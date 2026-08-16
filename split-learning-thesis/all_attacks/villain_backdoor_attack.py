import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config


class IndexedDataset(Dataset):

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        return image, label, index


def build_indexed_loader(base_dataset, batch_size=128, shuffle=True):
    return DataLoader(IndexedDataset(base_dataset), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=True)


def extract_targets(base_dataset):
    targets = getattr(base_dataset, 'targets', None)
    if targets is None:
        targets = [base_dataset[i][1] for i in range(len(base_dataset))]
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.asarray(targets)


class CandidateSelector(nn.Module):

    def __init__(self, embedding_dim, hidden_dim=128):
        super(CandidateSelector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embedding):
        return self.net(embedding).squeeze(1)


class VILLAINBackdoorAttack:

    def __init__(self, client_model, server_model, base_dataset, dataset=Config.DATASET,
                 num_classes=Config.NUM_CLASSES, target_label=0, beta=1.0, trigger_fraction=0.5,
                 dropout_keep=0.75, gamma_low=0.6, gamma_high=1.2, theta=None, theta_quantile=None, poison_rate=0.01,
                 candidates_per_batch=14, boosted_learning_rate=5e-3, attack_learning_rate=1e-3,
                 server_learning_rate=1e-3, group_size=16, selector_steps=50):

        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.num_classes = num_classes
        self.target_label = target_label

        self.beta = beta
        self.trigger_fraction = trigger_fraction
        self.dropout_keep = dropout_keep
        self.gamma_low = gamma_low
        self.gamma_high = gamma_high
        self.theta = theta
        self.theta_quantile = theta_quantile if theta_quantile is not None else 1.0 / float(num_classes)
        self.poison_rate = poison_rate
        self.candidates_per_batch = candidates_per_batch
        self.boosted_learning_rate = boosted_learning_rate
        self.attack_learning_rate = attack_learning_rate
        self.group_size = group_size
        self.selector_steps = selector_steps

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.in_channels = 1 if dataset == 'MNIST' else 3
        self.image_size = 28 if dataset == 'MNIST' else 32
        self._materialise_lazy_modules()

        self.embedding_dim = int(np.prod(self.smashed_shape[1:]))

        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=boosted_learning_rate)
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=server_learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.selector = CandidateSelector(self.embedding_dim).to(self.device)
        self.selector_optimizer = optim.Adam(self.selector.parameters(), lr=1e-3)
        self.selector_criterion = nn.BCEWithLogitsLoss()
        self.selector_features = []
        self.selector_labels = []

        self.targets = extract_targets(base_dataset)
        self.base_dataset = base_dataset

        known = int(np.where(self.targets == target_label)[0][0])
        self.known_target_index = known
        self.group_indices = [known]
        self.group_inputs = base_dataset[known][0].unsqueeze(0).to(self.device)

        self.gradient_norm_history = {}
        self.inferred_targets = set()
        self.rejected_samples = set()
        self.mu = float('inf')
        self.resolved_threshold = float('nan')

        self.trigger_mask = None
        self.trigger_values = None
        self.trigger = None
        self.delta = 0.0

        self.history = {'epoch': [], 'asr': [], 'cda': []}

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("   VILLAIN -- CLIENT-SIDE TRIGGER BACKDOOR ATTACK")
        print("=" * 60)
        print("  Reference       : Bai et al., USENIX Security 2023")
        print(f"  Dataset         : {dataset}")
        print(f"  Target label    : {target_label}")
        print(f"  Smashed shape   : {tuple(self.smashed_shape[1:])}")
        print(f"  Embedding dim   : {self.embedding_dim}")
        print(f"  Trigger beta    : {beta} | Trigger fraction: {trigger_fraction}")
        print(f"  Dropout keep    : {dropout_keep} | Shifting: [{gamma_low}, {gamma_high}]")
        print(f"  Poisoning rate  : {poison_rate}")
        print(f"  Theta           : {'adaptive' if theta is None else theta} | Quantile: {self.theta_quantile:.4f}")

    def _materialise_lazy_modules(self):
        dummy = torch.zeros(2, self.in_channels, self.image_size, self.image_size,
                            device=self.device)
        with torch.no_grad():
            smashed = self.client_model(dummy)
            self.server_model(smashed)
        self.smashed_shape = smashed.shape

    def _set_client_learning_rate(self, learning_rate):
        for group in self.client_optimizer.param_groups:
            group['lr'] = learning_rate

    def _split_step(self, inputs, labels, transform=None):
        self.client_optimizer.zero_grad()
        embedding = self.client_model(inputs)

        blocked = None
        upload = embedding
        if transform is not None:
            upload, blocked = transform(embedding)

        upload_server = upload.detach().requires_grad_(True)

        self.server_optimizer.zero_grad()
        outputs = self.server_model(upload_server)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.server_optimizer.step()

        returned_gradient = upload_server.grad.detach()
        gradient_norms = returned_gradient.flatten(1).norm(p=2, dim=1)

        client_gradient = returned_gradient.clone()
        if blocked is not None and blocked.numel() > 0:
            client_gradient[blocked] = 0.0

        upload.backward(client_gradient)
        self.client_optimizer.step()

        return loss.item(), gradient_norms.detach(), outputs.detach()

    def warmup(self, train_loader, epochs=5):
        print(f"\n  Phase 1 -- normal training with boosted attacker learning rate ({epochs} epoch(s))...")

        self._set_client_learning_rate(self.boosted_learning_rate)
        self.client_model.train()
        self.server_model.train()

        for epoch in range(epochs):
            running_loss, correct, total = 0.0, 0, 0

            progress = tqdm(train_loader, desc=f"  Warmup epoch [{epoch+1}/{epochs}]", leave=False)
            for inputs, labels, indices in progress:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                loss_value, gradient_norms, outputs = self._split_step(inputs, labels)

                for position in range(len(indices)):
                    self.gradient_norm_history[int(indices[position])] = float(gradient_norms[position])

                running_loss += loss_value
                total += labels.size(0)
                correct += outputs.argmax(1).eq(labels).sum().item()
                progress.set_postfix({'Loss': f'{loss_value:.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {running_loss/len(train_loader):.4f} | "
                  f"Train Acc: {100.*correct/total:.2f}%")

        values = np.asarray(list(self.gradient_norm_history.values()), dtype=float)
        self.mu = float(values.mean())
        print(f"  Gradient norm threshold mu = {self.mu:.6f}")

    def _sample_group_embeddings(self, count):
        choice = torch.randint(len(self.group_inputs), (count,), device=self.device)
        with torch.no_grad():
            return self.client_model(self.group_inputs[choice]).detach()

    def _extend_inference_group(self, index):
        if len(self.group_indices) >= self.group_size or index in self.group_indices:
            return
        image = self.base_dataset[index][0].unsqueeze(0).to(self.device)
        self.group_inputs = torch.cat([self.group_inputs, image], dim=0)
        self.group_indices.append(index)

    def _train_selector(self):
        if len(self.selector_features) == 0:
            return
        features = torch.cat(self.selector_features, dim=0)
        labels = torch.cat(self.selector_labels, dim=0)
        if labels.sum() == 0:
            return

        self.selector.train()
        for _ in range(self.selector_steps):
            self.selector_optimizer.zero_grad()
            loss = self.selector_criterion(self.selector(features), labels)
            loss.backward()
            self.selector_optimizer.step()

    def _resolve_inferences(self, pending):
        if len(pending) == 0:
            return 0

        admissible = [record for record in pending if record[2] <= self.mu]
        if len(admissible) == 0:
            return 0

        if self.theta is not None:
            threshold = self.theta
        else:
            ratios = np.asarray([record[1] for record in admissible], dtype=float)
            threshold = float(np.quantile(ratios, self.theta_quantile))

        accepted = 0
        for index, ratio, previous, feature in pending:
            if previous <= self.mu and ratio <= threshold:
                self.inferred_targets.add(index)
                self._extend_inference_group(index)
                self.selector_labels.append(torch.ones(1, device=self.device))
                accepted += 1
            else:
                self.rejected_samples.add(index)
                self.selector_labels.append(torch.zeros(1, device=self.device))
            self.selector_features.append(feature)

        self.resolved_threshold = threshold
        return accepted

    def infer_labels(self, train_loader, epochs=5):
        print(f"\n  Phase 2 -- label inference by embedding swapping ({epochs} epoch(s))...")

        self.client_model.train()
        self.server_model.train()

        for epoch in range(epochs):
            swapped_total = 0
            pending = []

            progress = tqdm(train_loader, desc=f"  Inference epoch [{epoch+1}/{epochs}]", leave=False)
            for inputs, labels, indices in progress:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                indices = indices.numpy()

                with torch.no_grad():
                    preview = self.client_model(inputs).flatten(1)
                    scores = self.selector(preview)

                eligible = [position for position in range(len(indices))
                            if int(indices[position]) in self.gradient_norm_history
                            and int(indices[position]) not in self.inferred_targets
                            and int(indices[position]) not in self.rejected_samples]

                if len(eligible) == 0:
                    self._split_step(inputs, labels)
                    continue

                eligible = torch.tensor(eligible, device=self.device)
                ranking = torch.argsort(scores[eligible], descending=True)
                candidates = eligible[ranking[:self.candidates_per_batch]]

                swap_source = self._sample_group_embeddings(len(candidates))

                def transform(embedding):
                    uploaded = embedding.clone()
                    uploaded[candidates] = swap_source
                    return uploaded, candidates

                loss_value, gradient_norms, _ = self._split_step(inputs, labels, transform)

                candidate_set = set(candidates.cpu().numpy().tolist())

                for position in range(len(indices)):
                    index = int(indices[position])
                    if position in candidate_set:
                        continue
                    self.gradient_norm_history[index] = float(gradient_norms[position])

                for position in candidates.cpu().numpy().tolist():
                    index = int(indices[position])
                    previous = self.gradient_norm_history[index]
                    ratio = float(gradient_norms[position]) / (previous + 1e-12)
                    pending.append((index, ratio, previous, preview[position].unsqueeze(0).detach()))

                swapped_total += len(candidates)
                progress.set_postfix({'Loss': f'{loss_value:.4f}',
                                      'Pending': len(pending)})

            values = np.asarray(list(self.gradient_norm_history.values()), dtype=float)
            self.mu = float(values.mean())

            accepted = self._resolve_inferences(pending)
            self._train_selector()

            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Swapped: {swapped_total} | "
                  f"Accepted: {accepted} | "
                  f"Inferred target samples: {len(self.inferred_targets)} | "
                  f"theta: {self.resolved_threshold:.4f} | "
                  f"mu: {self.mu:.6f}")

        return self.label_inference_accuracy()

    def label_inference_accuracy(self):
        if len(self.inferred_targets) == 0:
            return 0.0
        inferred = np.asarray(sorted(self.inferred_targets))
        return float(np.mean(self.targets[inferred] == self.target_label))

    def fabricate_trigger(self, train_loader, batches=20):
        print("\n  Phase 3 -- trigger fabrication on the embedding vector...")

        self.client_model.eval()
        collected = []

        with torch.no_grad():
            for batch_number, (inputs, labels, indices) in enumerate(train_loader):
                if batch_number >= batches:
                    break
                collected.append(self.client_model(inputs.to(self.device)).flatten(1))

        embeddings = torch.cat(collected, dim=0)
        deviation = embeddings.std(dim=0)

        length = max(4, int(self.trigger_fraction * self.embedding_dim))
        area = torch.topk(deviation, length).indices.sort().values

        self.delta = float(deviation[area].mean())

        signs = torch.tensor([1.0, 1.0, -1.0, -1.0], device=self.device).repeat(length // 4 + 1)[:length]

        self.trigger_mask = torch.zeros(self.embedding_dim, device=self.device)
        self.trigger_mask[area] = 1.0

        self.trigger_values = torch.zeros(self.embedding_dim, device=self.device)
        self.trigger_values[area] = self.beta * self.delta * signs

        self.trigger = self.trigger_mask * self.trigger_values

        print(f"  Trigger length : {length} of {self.embedding_dim} elements")
        print(f"  Trigger delta  : {self.delta:.6f}")
        print(f"  Trigger norm   : {float(self.trigger.norm(p=2)):.6f}")

    def _augmented_trigger(self):
        dropout = (torch.rand(self.embedding_dim, device=self.device) < self.dropout_keep).float()
        gamma = float(np.random.uniform(self.gamma_low, self.gamma_high))
        return self.trigger_mask * dropout * gamma * self.trigger_values

    def inject_backdoor(self, train_loader, test_loader, epochs=10):
        print(f"\n  Phase 4 -- backdoor injection with additive embedding trigger ({epochs} epoch(s))...")

        self._set_client_learning_rate(self.attack_learning_rate)
        self.client_model.train()
        self.server_model.train()

        for epoch in range(epochs):
            poisoned_total = 0

            progress = tqdm(train_loader, desc=f"  Injection epoch [{epoch+1}/{epochs}]", leave=False)
            for inputs, labels, indices in progress:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                budget = max(1, int(self.poison_rate * len(indices)))
                available = [position for position in range(len(indices))
                             if int(indices[position]) in self.inferred_targets]
                poisoned = available[:budget]

                if len(poisoned) == 0:
                    loss_value, _, _ = self._split_step(inputs, labels)
                    progress.set_postfix({'Loss': f'{loss_value:.4f}'})
                    continue

                rows = torch.tensor(poisoned, device=self.device)
                trigger = self._augmented_trigger()

                def transform(embedding):
                    flattened = embedding.flatten(1)
                    perturbation = torch.zeros_like(flattened)
                    perturbation[rows] = trigger
                    uploaded = (flattened + perturbation).view_as(embedding)
                    return uploaded, None

                loss_value, _, _ = self._split_step(inputs, labels, transform)
                poisoned_total += len(poisoned)
                progress.set_postfix({'Loss': f'{loss_value:.4f}', 'Poisoned': poisoned_total})

            clean_accuracy, attack_success = self.evaluate(test_loader)
            self.history['epoch'].append(epoch + 1)
            self.history['asr'].append(attack_success)
            self.history['cda'].append(clean_accuracy)

            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Poisoned uploads: {poisoned_total} | "
                  f"ASR: {attack_success:.2f}% | "
                  f"CDA: {clean_accuracy:.2f}%")

    def evaluate(self, test_loader):
        self.client_model.eval()
        self.server_model.eval()

        clean_correct, clean_total = 0, 0
        attack_success, attack_total = 0, 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                embedding = self.client_model(inputs)
                outputs = self.server_model(embedding)
                clean_total += labels.size(0)
                clean_correct += outputs.argmax(1).eq(labels).sum().item()

                if self.trigger is None:
                    continue

                rows = labels != self.target_label
                if rows.sum() == 0:
                    continue

                triggered = (embedding.flatten(1)[rows] + self.trigger).view(-1, *embedding.shape[1:])
                predictions = self.server_model(triggered).argmax(1)
                attack_total += int(rows.sum())
                attack_success += int(predictions.eq(self.target_label).sum())

        self.client_model.train()
        self.server_model.train()

        clean_accuracy = 100.0 * clean_correct / clean_total
        success_rate = 100.0 * attack_success / attack_total if attack_total > 0 else 0.0

        return clean_accuracy, success_rate

    def summarise(self, test_loader=None, clean_baseline=None):
        if test_loader is not None:
            clean_accuracy, attack_success = self.evaluate(test_loader)
        else:
            clean_accuracy, attack_success = self.history['cda'][-1], self.history['asr'][-1]

        summary = {'target_label': self.target_label,
                   'lia': 100.0 * self.label_inference_accuracy(),
                   'asr': attack_success,
                   'cda': clean_accuracy,
                   'inferred_samples': len(self.inferred_targets),
                   'trigger_delta': self.delta}

        if clean_baseline is not None:
            summary['clean_baseline'] = clean_baseline

        print("\n" + "=" * 60)
        print("   VILLAIN BACKDOOR RESULTS")
        print("=" * 60)
        print(f"  Label inference accuracy (LIA) : {summary['lia']:.2f}%")
        print(f"  Attack success rate (ASR)      : {summary['asr']:.2f}%")
        print(f"  Clean data accuracy (CDA)      : {summary['cda']:.2f}%")
        print(f"  Inferred target samples        : {summary['inferred_samples']}")
        print("=" * 60)

        return summary

    def save_visualization(self, tag='no_defense'):
        if len(self.history['epoch']) == 0:
            return

        plt.figure(figsize=(7, 5))
        plt.plot(self.history['epoch'], self.history['asr'], marker='o', linewidth=2, label='ASR')
        plt.plot(self.history['epoch'], self.history['cda'], marker='s', linewidth=2, label='CDA')
        plt.xlabel('Injection Epoch')
        plt.ylabel('Percentage (%)')
        plt.ylim(0, 105)
        plt.title(f'VILLAIN -- {tag.replace("_", " ").title()}\n'
                  f'{self.dataset} | Target Label {self.target_label} | Cut Layer {Config.CUT_LAYER}',
                  fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = f"{Config.RESULTS_DIR}/villain_{tag}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved -> {save_path}")