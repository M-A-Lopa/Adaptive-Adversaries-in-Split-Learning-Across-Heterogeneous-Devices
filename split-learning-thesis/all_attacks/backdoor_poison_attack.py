import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config


def _get_norm_stats(dataset, device):
    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010], device=device).view(1, 3, 1, 1)
    else:
        mean = torch.tensor([0.1307], device=device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081], device=device).view(1, 1, 1, 1)
    return mean, std


def denormalize(x, dataset):
    mean, std = _get_norm_stats(dataset, x.device)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def normalize(x, dataset):
    mean, std = _get_norm_stats(dataset, x.device)
    return (x - mean) / std


def apply_trigger_patch(images_raw01, patch_size=4, value=1.0):
    """BadNets-style fixed pixel patch in the bottom-right corner (Chen et al., 2017)."""
    patched = images_raw01.clone()
    h, w = patched.shape[-2], patched.shape[-1]
    patched[:, :, h - patch_size:h, w - patch_size:w] = value
    return patched


def extract_targets(base_dataset):
    targets = getattr(base_dataset, 'targets', None)
    if targets is None:
        targets = [base_dataset[i][1] for i in range(len(base_dataset))]
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.asarray(targets)


class BackdoorPoisonAttack:

    def __init__(self, client_model, server_model, base_dataset, dataset=Config.DATASET,
                 num_classes=Config.NUM_CLASSES, mode='client', target_label=0,
                 poison_rate=0.05, patch_size=4, trigger_value=1.0,
                 surrogate_builder=None, learning_rate=1e-3):

        assert mode in ('client', 'server'), "mode must be 'client' or 'server'"

        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.num_classes = num_classes
        self.mode = mode
        self.target_label = target_label
        self.poison_rate = poison_rate
        self.patch_size = patch_size
        self.trigger_value = trigger_value

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=learning_rate)
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.targets = extract_targets(base_dataset)

        self.surrogate_client = None
        self.surrogate_optimizer = None
        if mode == 'server':
            if surrogate_builder is None:
                raise ValueError(
                    "mode='server' requires a surrogate_builder callable that "
                    "returns a fresh client-architecture instance."
                )
            self.surrogate_client = surrogate_builder().to(self.device)
            self.surrogate_optimizer = optim.Adam(self.surrogate_client.parameters(), lr=learning_rate)

        self.history = {'epoch': [], 'asr': [], 'cda': []}

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("   BACKDOOR POISONING ATTACK -- MODEL INTEGRITY")
        print("=" * 60)
        print(f"  Mode            : {mode}")
        ref = "Chen et al., arXiv 2017 [25]" if mode == 'client' \
              else "Tajalli et al., IEEE SPW 2023 [23] (surrogate client)"
        print(f"  Reference       : {ref}")
        print(f"  Dataset         : {dataset}")
        print(f"  Target label    : {target_label}")
        print(f"  Poison rate     : {poison_rate}")
        print(f"  Trigger         : {patch_size}x{patch_size} patch, value={trigger_value} "
              f"(BadNets-style, bottom-right corner)")

    def _poison_batch(self, images, labels):

        labels = labels.clone()
        eligible = (labels != self.target_label).nonzero(as_tuple=True)[0]
        if eligible.numel() == 0:
            return images, labels

        n_poison = max(1, int(self.poison_rate * images.shape[0]))
        n_poison = min(n_poison, eligible.numel())
        perm = torch.randperm(eligible.numel(), device=images.device)[:n_poison]
        chosen = eligible[perm]

        raw = denormalize(images, self.dataset)
        raw[chosen] = apply_trigger_patch(raw[chosen], self.patch_size, self.trigger_value)
        images = normalize(raw, self.dataset)
        labels[chosen] = self.target_label

        return images, labels

    def _apply_trigger_eval(self, images):
        raw = denormalize(images, self.dataset)
        raw = apply_trigger_patch(raw, self.patch_size, self.trigger_value)
        return normalize(raw, self.dataset)

    # ---------------------------------------------------- server pre-poisoning

    def pretrain_server_backdoor(self, auxiliary_loader, epochs=5):

        self.surrogate_client.train()
        self.server_model.train()

        for epoch in range(epochs):
            running_loss, correct, total = 0.0, 0, 0
            progress = tqdm(auxiliary_loader, desc=f"  Surrogate epoch [{epoch+1}/{epochs}]", leave=False)

            for images, labels in progress:
                images = images.to(self.device)
                labels = labels.to(self.device)

                images_p, labels_p = self._poison_batch(images, labels)

                self.surrogate_optimizer.zero_grad()
                self.server_optimizer.zero_grad()

                smashed = self.surrogate_client(images_p)
                outputs = self.server_model(smashed)
                loss = self.criterion(outputs, labels_p)
                loss.backward()

                self.surrogate_optimizer.step()
                self.server_optimizer.step()

                running_loss += loss.item()
                total += labels_p.size(0)
                correct += outputs.argmax(1).eq(labels_p).sum().item()
                progress.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.1f}%'})

            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {running_loss/len(auxiliary_loader):.4f} | "
                  f"Surrogate Acc: {100.*correct/total:.2f}%")

        print("  Server half is now pre-poisoned. Handing off to legitimate "
              "split-learning training with the honest client...")


    def _split_step(self, inputs, labels):
        self.client_optimizer.zero_grad()
        embedding = self.client_model(inputs)

        upload = embedding.detach().requires_grad_(True)

        self.server_optimizer.zero_grad()
        outputs = self.server_model(upload)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.server_optimizer.step()

        embedding.backward(upload.grad)
        self.client_optimizer.step()

        return loss.item(), outputs.detach()

    def train(self, train_loader, test_loader, epochs=10):

        print(f"\n  Phase 2 -- split-learning training ({epochs} epoch(s), "
              f"client is {'malicious' if self.mode == 'client' else 'honest'})...")

        self.client_model.train()
        self.server_model.train()

        for epoch in range(epochs):
            running_loss, correct, total = 0.0, 0, 0
            progress = tqdm(train_loader, desc=f"  Poison epoch [{epoch+1}/{epochs}]", leave=False)

            for images, labels in progress:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if self.mode == 'client':
                    images, labels = self._poison_batch(images, labels)

                loss_value, outputs = self._split_step(images, labels)

                running_loss += loss_value
                total += labels.size(0)
                correct += outputs.argmax(1).eq(labels).sum().item()
                progress.set_postfix({'Loss': f'{loss_value:.4f}', 'Acc': f'{100.*correct/total:.1f}%'})

            clean_acc, attack_success = self.evaluate(test_loader)
            self.history['epoch'].append(epoch + 1)
            self.history['asr'].append(attack_success)
            self.history['cda'].append(clean_acc)

            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {running_loss/len(train_loader):.4f} | "
                  f"ASR: {attack_success:.2f}% | CDA: {clean_acc:.2f}%")


    def evaluate(self, test_loader):
        self.client_model.eval()
        self.server_model.eval()

        clean_correct, clean_total = 0, 0
        attack_success, attack_total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.server_model(self.client_model(images))
                clean_total += labels.size(0)
                clean_correct += outputs.argmax(1).eq(labels).sum().item()

                rows = labels != self.target_label
                if rows.sum() == 0:
                    continue

                triggered = self._apply_trigger_eval(images[rows])
                triggered_outputs = self.server_model(self.client_model(triggered))
                attack_total += int(rows.sum())
                attack_success += int(triggered_outputs.argmax(1).eq(self.target_label).sum())

        self.client_model.train()
        self.server_model.train()

        clean_accuracy = 100.0 * clean_correct / clean_total
        success_rate = 100.0 * attack_success / attack_total if attack_total > 0 else 0.0
        return clean_accuracy, success_rate

    def summarise(self):
        clean_acc = self.history['cda'][-1] if self.history['cda'] else float('nan')
        asr = self.history['asr'][-1] if self.history['asr'] else float('nan')

        summary = {
            'mode': self.mode,
            'target_label': self.target_label,
            'poison_rate': self.poison_rate,
            'asr': asr,
            'cda': clean_acc,
        }

        print("\n" + "=" * 60)
        print(f"   BACKDOOR POISONING RESULTS -- mode='{self.mode}'")
        print("=" * 60)
        print(f"  Attack success rate (ASR) : {asr:.2f}%")
        print(f"  Clean data accuracy (CDA) : {clean_acc:.2f}%")
        print("=" * 60)

        return summary

    def save_visualization(self, tag='no_defense'):
        if len(self.history['epoch']) == 0:
            return

        plt.figure(figsize=(7, 5))
        plt.plot(self.history['epoch'], self.history['asr'], marker='o', linewidth=2, label='ASR')
        plt.plot(self.history['epoch'], self.history['cda'], marker='s', linewidth=2, label='CDA')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 105)
        plt.title(f"Backdoor Poisoning ({self.mode}) -- {tag.replace('_', ' ').title()}\n"
                  f"{self.dataset} | Target Label {self.target_label} | Cut Layer {Config.CUT_LAYER}",
                  fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = f"{Config.RESULTS_DIR}/backdoor_poison_{self.mode}_{tag}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved -> {save_path}")