import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from config import Config


class BinaryImbalancedView(Dataset):

    def __init__(self, base_dataset, indices, labels):
        self.base_dataset = base_dataset
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, _ = self.base_dataset[int(self.indices[position])]
        return image, float(self.labels[position])


def extract_targets(base_dataset):
    targets = getattr(base_dataset, 'targets', None)
    if targets is None:
        targets = [base_dataset[i][1] for i in range(len(base_dataset))]
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    return np.asarray(targets)


def build_binary_split_loaders(base_dataset, target_class=0, positive_ratio=0.1,
                               batch_size=128, train_fraction=0.8, seed=0):
    targets = extract_targets(base_dataset)
    rng = np.random.RandomState(seed)

    positive_pool = np.where(targets == target_class)[0]
    negative_pool = np.where(targets != target_class)[0]

    negative_count = int(round(len(positive_pool) * (1.0 - positive_ratio) / positive_ratio))
    negative_count = min(negative_count, len(negative_pool))
    negative_pool = rng.choice(negative_pool, size=negative_count, replace=False)

    selected = np.concatenate([positive_pool, negative_pool])
    rng.shuffle(selected)

    labels = (targets[selected] == target_class).astype(np.float32)
    cut = int(len(selected) * train_fraction)

    train_view = BinaryImbalancedView(base_dataset, selected[:cut], labels[:cut])
    test_view = BinaryImbalancedView(base_dataset, selected[cut:], labels[cut:])

    train_loader = DataLoader(train_view, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_view, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"  Positive class    : {target_class}")
    print(f"  Positive fraction : {100.0 * float(labels.mean()):.2f}%")
    print(f"  Train / Test size : {len(train_view)} / {len(test_view)}")

    return train_loader, test_loader


def first_client_block(client_model):
    children = list(client_model.children())
    head = children[0]
    inner = list(head.children())
    if len(inner) > 0 and isinstance(inner[0], nn.Sequential):
        return inner[0]
    return head


def norm_scoring_function(gradients):
    return gradients.norm(p=2, dim=1)


def direction_scoring_function(gradients, positive_gradient):
    return F.cosine_similarity(gradients, positive_gradient.unsqueeze(0), dim=1)


def leak_auc(scores, labels):
    labels = labels.detach().cpu().numpy()
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores.detach().cpu().numpy()))


def majority_counting_accuracy(gradients, labels):
    normalised = F.normalize(gradients, p=2, dim=1)
    similarity = normalised @ normalised.t()
    agreement = (similarity > 0).float().mean(dim=1)
    predicted = (agreement <= 0.5).float()
    return float((predicted == labels).float().mean().item())


class GradientNormLabelLeakageAttack:

    def __init__(self, client_model, server_model, dataset=Config.DATASET,
                 target_class=0, learning_rate=1e-4):

        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.target_class = target_class

        self.client_model = client_model.to(self.device)
        self.server_model = server_model.to(self.device)

        self.in_channels = 1 if dataset == 'MNIST' else 3
        self.image_size = 28 if dataset == 'MNIST' else 32
        self._materialise_lazy_modules()

        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=learning_rate)
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.first_activation = None
        first_client_block(self.client_model).register_forward_hook(self._capture_first_activation)

        self.history = {'iteration': [], 'train_loss': [],
                        'norm_leak_auc_cut': [], 'cosine_leak_auc_cut': [],
                        'norm_leak_auc_first': [], 'cosine_leak_auc_first': [],
                        'majority_accuracy_cut': []}
        self.test_auc = float('nan')

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("   LABEL LEAKAGE -- NORM AND DIRECTION SCORING ATTACKS")
        print("=" * 60)
        print("  Reference     : Li et al., ICLR 2022")
        print(f"  Dataset       : {dataset}")
        print(f"  Cut layer     : {Config.CUT_LAYER}")
        print(f"  Smashed shape : {tuple(self.smashed_shape[1:])}")
        print(f"  Learning rate : {learning_rate}")

    def _materialise_lazy_modules(self):
        dummy = torch.zeros(2, self.in_channels, self.image_size, self.image_size,
                            device=self.device)
        with torch.no_grad():
            smashed = self.client_model(dummy)
            self.server_model(smashed)
        self.smashed_shape = smashed.shape

    def _capture_first_activation(self, module, inputs, output):
        if output.requires_grad:
            output.retain_grad()
        self.first_activation = output

    def _record(self, iteration, loss_value, cut_gradient, first_gradient, labels):
        positive_positions = torch.nonzero(labels > 0.5).flatten()
        if positive_positions.numel() == 0:
            return

        reference = positive_positions[torch.randint(len(positive_positions), (1,))].item()

        norm_cut = leak_auc(norm_scoring_function(cut_gradient), labels)
        cosine_cut = leak_auc(direction_scoring_function(cut_gradient, cut_gradient[reference]), labels)
        norm_first = leak_auc(norm_scoring_function(first_gradient), labels)
        cosine_first = leak_auc(direction_scoring_function(first_gradient, first_gradient[reference]), labels)

        if norm_cut is None:
            return

        self.history['iteration'].append(iteration)
        self.history['train_loss'].append(loss_value)
        self.history['norm_leak_auc_cut'].append(norm_cut)
        self.history['cosine_leak_auc_cut'].append(cosine_cut)
        self.history['norm_leak_auc_first'].append(norm_first)
        self.history['cosine_leak_auc_first'].append(cosine_first)
        self.history['majority_accuracy_cut'].append(majority_counting_accuracy(cut_gradient, labels))

    def run(self, train_loader, test_loader, epochs=5):
        print(f"\n  Running split training and per-batch label leakage measurement for {epochs} epoch(s)...")

        iteration = 0

        for epoch in range(epochs):
            self.client_model.train()
            self.server_model.train()

            progress = tqdm(train_loader, desc=f"  Leakage epoch [{epoch+1}/{epochs}]", leave=False)
            for inputs, labels in progress:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float()

                self.client_optimizer.zero_grad()
                smashed = self.client_model(inputs)

                smashed_server = smashed.detach().requires_grad_(True)

                self.server_optimizer.zero_grad()
                logits = self.server_model(smashed_server).squeeze(1)
                loss = self.criterion(logits, labels)

                loss.backward()
                self.server_optimizer.step()

                cut_gradient = smashed_server.grad.detach().flatten(1)

                smashed.backward(smashed_server.grad)
                self.client_optimizer.step()

                first_gradient = self.first_activation.grad.detach().flatten(1)

                self._record(iteration, loss.item(), cut_gradient, first_gradient, labels)
                iteration += 1

                if len(self.history['iteration']) > 0:
                    progress.set_postfix({'Loss': f'{loss.item():.4f}',
                                          'NormAUC': f"{self.history['norm_leak_auc_cut'][-1]:.3f}",
                                          'CosAUC': f"{self.history['cosine_leak_auc_cut'][-1]:.3f}"})

            self.test_auc = self.evaluate(test_loader)
            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {self.history['train_loss'][-1]:.4f} | "
                  f"Norm leak AUC: {self.history['norm_leak_auc_cut'][-1]:.4f} | "
                  f"Cosine leak AUC: {self.history['cosine_leak_auc_cut'][-1]:.4f} | "
                  f"Test AUC: {self.test_auc:.4f}")

        return self.summarise()

    def evaluate(self, test_loader):
        self.client_model.eval()
        self.server_model.eval()

        scores = []
        truths = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                logits = self.server_model(self.client_model(inputs)).squeeze(1)
                scores.append(logits.cpu().numpy())
                truths.append(labels.numpy())

        scores = np.concatenate(scores)
        truths = np.concatenate(truths)

        if truths.min() == truths.max():
            return float('nan')
        return float(roc_auc_score(truths, scores))

    def summarise(self):
        summary = {'target_class': self.target_class, 'test_auc': self.test_auc}

        for key in ['norm_leak_auc_cut', 'cosine_leak_auc_cut',
                    'norm_leak_auc_first', 'cosine_leak_auc_first',
                    'majority_accuracy_cut']:
            values = np.asarray(self.history[key], dtype=float)
            summary[f'mean_{key}'] = float(np.mean(values))
            summary[f'q95_{key}'] = float(np.quantile(values, 0.95))

        print("\n" + "=" * 60)
        print("   LABEL LEAKAGE RESULTS (95% QUANTILE OVER BATCHES)")
        print("=" * 60)
        print(f"  Norm leak AUC   [cut layer]   : {summary['q95_norm_leak_auc_cut']:.4f}")
        print(f"  Cosine leak AUC [cut layer]   : {summary['q95_cosine_leak_auc_cut']:.4f}")
        print(f"  Norm leak AUC   [first layer] : {summary['q95_norm_leak_auc_first']:.4f}")
        print(f"  Cosine leak AUC [first layer] : {summary['q95_cosine_leak_auc_first']:.4f}")
        print(f"  Majority counting accuracy    : {summary['q95_majority_accuracy_cut']:.4f}")
        print(f"  Test AUC (utility)            : {summary['test_auc']:.4f}")
        print("=" * 60)

        return summary

    def save_visualization(self, tag='no_defense'):
        if len(self.history['iteration']) == 0:
            return

        iterations = self.history['iteration']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ax1.plot(iterations, self.history['norm_leak_auc_cut'], linewidth=1.5, label='Norm')
        ax1.plot(iterations, self.history['cosine_leak_auc_cut'], linewidth=1.5, label='Cosine')
        ax1.axhline(0.5, color='k', linestyle='--', linewidth=1)
        ax1.set_title('Leak AUC at Cut Layer', fontsize=13)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Leak AUC')
        ax1.set_ylim(0.4, 1.02)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(iterations, self.history['norm_leak_auc_first'], linewidth=1.5, label='Norm')
        ax2.plot(iterations, self.history['cosine_leak_auc_first'], linewidth=1.5, label='Cosine')
        ax2.axhline(0.5, color='k', linestyle='--', linewidth=1)
        ax2.set_title('Leak AUC at First Layer', fontsize=13)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Leak AUC')
        ax2.set_ylim(0.4, 1.02)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f'Label Leakage -- {tag.replace("_", " ").title()}\n'
                     f'{self.dataset} | Positive Class {self.target_class} | Cut Layer {Config.CUT_LAYER}',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = f"{Config.RESULTS_DIR}/label_leakage_{tag}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved -> {save_path}")