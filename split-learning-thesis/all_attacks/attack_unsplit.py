import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from config import Config
from all_model.models import ClientModel

def total_variation(x):

    diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    return diff_h.mean() + diff_w.mean()

def denormalize(tensor, dataset='CIFAR10'):

    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
        
    else: 
        mean = torch.tensor([0.1307]).view(1, 1, 1).to(tensor.device)
        std  = torch.tensor([0.3081]).view(1, 1, 1).to(tensor.device)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)

import torch
import numpy as np

def compute_mse(original, reconstructed):

    return torch.mean((original - reconstructed) ** 2).item()


def compute_psnr(original, reconstructed):

    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:

        return float('inf')
    
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def compute_ssim(original, reconstructed):

    if original.dim() == 4:
        scores = [_ssim_single(original[i], reconstructed[i])
                  for i in range(original.shape[0])]
        
        return float(np.mean(scores))
    
    return _ssim_single(original, reconstructed)


def _ssim_single(x, y):

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x    = x.mean().item()
    mu_y    = y.mean().item()
    sig_x   = x.var().item()
    sig_y   = y.var().item()
    sig_xy  = ((x - mu_x) * (y - mu_y)).mean().item()

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    denominator = (mu_x**2 + mu_y**2 + C1) * (sig_x + sig_y + C2)

    return numerator / (denominator + 1e-8)


def compute_accuracy(model_client, model_server, data_loader, device):

    model_client.eval()
    model_server.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            smashed = model_client(images)
            outputs = model_server(smashed)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def compute_accuracy_with_defense(model_client, model_server, defense_fn, data_loader, device):

    model_client.eval()
    model_server.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images  = images.to(device)
            labels  = labels.to(device)
            smashed = model_client(images)

            smashed_protected = defense_fn(smashed)

            outputs = model_server(smashed_protected)
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


def print_metrics_table(results: dict):

    header = f"\n{'Method':<20} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>10} {'Accuracy':>12}"
    print("\n" + "=" * 68)
    print("   DEFENSE COMPARISON TABLE")
    print("=" * 68)
    print(header)
    print("-" * 68)

    for method, m in results.items():
        print(f"  {method:<18} {m['mse']:>10.5f} {m['psnr']:>12.2f} "
              f"{m['ssim']:>10.4f} {m['accuracy']:>11.2f}%")
        
    print("=" * 68)
    print("\n  Interpretation:")
    print("  Lower MSE/PSNR/SSIM = stronger defense (harder to reconstruct)")
    print("  Higher Accuracy      = better utility preservation")
    
class UnSplitAttack:

    def __init__(self, client_model, in_channels=3):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

        self.client_model = client_model.to(self.device)
        self.client_model.eval()

        self.clone_model = ClientModel(in_channels=in_channels).to(self.device)

        self.tv_lambda = 1e-3

        self.clone_optimizer = optim.Adam(self.clone_model.parameters(), lr=0.001)

        self.mse_loss = nn.MSELoss()

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

        print("\n" + "="*60)
        print("   UNSPLIT ATTACK — INITIALISED")
        print("="*60)
        print(f"  Attack type    : Data-Oblivious Model Inversion")
        print(f"  Requires data  : NO (architecture knowledge only)")
        print(f"  TV lambda      : {self.tv_lambda}")
        print(f"  Clone params   : "
              f"{sum(p.numel() for p in self.clone_model.parameters()):,}")

    def _reconstruct_batch(self, smashed_data, input_shape, inversion_steps=300, lr_input=0.1):
        
        batch_size = smashed_data.shape[0]

        self.clone_model.train()
        for _ in range(10):  
            self.clone_optimizer.zero_grad()

            dummy = torch.randn(batch_size, *input_shape, device=self.device, requires_grad=False)
            clone_smashed = self.clone_model(dummy)

            loss_steal = (self.mse_loss(clone_smashed.mean(dim=0), smashed_data.mean(dim=0)) + self.mse_loss(clone_smashed.std(dim=0),  smashed_data.std(dim=0)))
            loss_steal.backward()
            self.clone_optimizer.step()

        self.clone_model.eval()

        dummy_input = torch.rand(batch_size, *input_shape,device=self.device).requires_grad_(True)

        input_optimizer = optim.Adam([dummy_input], lr=lr_input)

        best_loss    = float('inf')
        best_dummy   = dummy_input.detach().clone()

        for step in range(inversion_steps):
            input_optimizer.zero_grad()

            clone_output = self.clone_model(dummy_input)

            loss_inv = self.mse_loss(clone_output, smashed_data.detach())

            loss_tv  = self.tv_lambda * total_variation(dummy_input)

            loss = loss_inv + loss_tv
            loss.backward()
            input_optimizer.step()

            with torch.no_grad():
                dummy_input.clamp_(0.0, 1.0)

            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_dummy = dummy_input.detach().clone()

        return best_dummy

    def run_attack(self, data_loader, num_batches=20, inversion_steps=300):
        
        print(f"\n  Running UnSplit attack on {num_batches} batches...")
        print(f"  Inversion steps per batch: {inversion_steps}")

        in_channels = 1 if Config.DATASET == 'MNIST' else 3
        if Config.DATASET == 'MNIST':
            input_shape = (1, 28, 28)
        else:
            input_shape = (3, 32, 32)

        all_psnr  = []
        all_ssim  = []
        all_mse   = []

        originals_store      = []
        reconstructed_store  = []

        for batch_idx, (images, _) in enumerate(tqdm(data_loader, total=num_batches, desc="  Attacking batches")):

            if batch_idx >= num_batches:
                break

            images = images.to(self.device)

            with torch.no_grad():
                smashed_data = self.client_model(images)

            reconstructed = self._reconstruct_batch(smashed_data, input_shape, inversion_steps)

            originals_dn = denormalize(images, Config.DATASET)

            for i in range(images.shape[0]):
                orig = originals_dn[i]
                rec  = reconstructed[i].clamp(0, 1)
                all_psnr.append(compute_psnr(orig, rec))
                all_ssim.append(compute_ssim(orig.unsqueeze(0), rec.unsqueeze(0)))
                all_mse.append(compute_mse(orig, rec))

            if batch_idx == 0:
                originals_store     = originals_dn[:8].cpu()
                reconstructed_store = reconstructed[:8].cpu()

        mean_psnr = float(np.mean(all_psnr))
        mean_ssim = float(np.mean(all_ssim))
        mean_mse  = float(np.mean(all_mse))

        print("\n" + "="*60)
        print("   UNSPLIT ATTACK — RESULTS (NO DEFENSE)")
        print("="*60)
        print(f"  MSE  : {mean_mse:.5f}")
        print(f"  PSNR : {mean_psnr:.2f} dB")
        print(f"  SSIM : {mean_ssim:.4f}")
        print("="*60)
        print("  These are your BASELINE attack numbers.")
        print("  After defenses, all three metrics should improve.")

        self._save_visualization(originals_store, reconstructed_store, tag='no_defense')

        return {'mse' : mean_mse, 'psnr': mean_psnr, 'ssim': mean_ssim}

    def run_attack_with_defense(self, data_loader, defense_fn, defense_name, num_batches=20, inversion_steps=300):

        print(f"\n  Running UnSplit attack WITH defense: {defense_name}")

        in_channels = 1 if Config.DATASET == 'MNIST' else 3
        if Config.DATASET == 'MNIST':
            input_shape = (1, 28, 28)
        else:
            input_shape = (3, 32, 32)

        all_psnr = []
        all_ssim = []
        all_mse  = []

        originals_store     = []
        reconstructed_store = []

        for batch_idx, (images, _) in enumerate(tqdm(data_loader, total=num_batches,desc=f"  Attacking [{defense_name}]")):

            if batch_idx >= num_batches:
                break

            images = images.to(self.device)

            with torch.no_grad():
                smashed_data = self.client_model(images)
                smashed_protected = defense_fn(smashed_data)

            reconstructed = self._reconstruct_batch(smashed_protected, input_shape, inversion_steps)

            originals_dn = denormalize(images, Config.DATASET)

            for i in range(images.shape[0]):
                orig = originals_dn[i]
                rec  = reconstructed[i].clamp(0, 1)
                all_psnr.append(compute_psnr(orig, rec))
                all_ssim.append(compute_ssim(orig.unsqueeze(0),
                                              rec.unsqueeze(0)))
                all_mse.append(compute_mse(orig, rec))

            if batch_idx == 0:
                originals_store     = originals_dn[:8].cpu()
                reconstructed_store = reconstructed[:8].cpu()

        mean_psnr = float(np.mean(all_psnr))
        mean_ssim = float(np.mean(all_ssim))
        mean_mse  = float(np.mean(all_mse))

        print(f"  MSE  : {mean_mse:.5f}")
        print(f"  PSNR : {mean_psnr:.2f} dB")
        print(f"  SSIM : {mean_ssim:.4f}")

        self._save_visualization(originals_store, reconstructed_store, tag=defense_name.lower().replace(' ', '_'))

        return {'mse' : mean_mse, 'psnr': mean_psnr, 'ssim': mean_ssim}

    def _save_visualization(self, originals, reconstructed, tag='result'):

        num = min(8, len(originals))
        fig = plt.figure(figsize=(num * 2, 5))
        gs  = gridspec.GridSpec(2, num, hspace=0.3)

        for i in range(num):
            ax1 = fig.add_subplot(gs[0, i])
            img = originals[i].permute(1, 2, 0).numpy()
            if img.shape[2] == 1:
                img = img.squeeze(2)
                ax1.imshow(img, cmap='gray')

            else:
                ax1.imshow(np.clip(img, 0, 1))
            ax1.axis('off')

            if i == 0:
                ax1.set_title('Original', fontsize=10, fontweight='bold')

            ax2 = fig.add_subplot(gs[1, i])
            rec = reconstructed[i].permute(1, 2, 0).numpy()

            if rec.shape[2] == 1:
                rec = rec.squeeze(2)
                ax2.imshow(rec, cmap='gray')

            else:
                ax2.imshow(np.clip(rec, 0, 1))
            ax2.axis('off')

            if i == 0:
                ax2.set_title('Reconstructed\n(Attacker)', fontsize=10,
                               fontweight='bold')

        plt.suptitle(f'UnSplit Attack — {tag.replace("_", " ").title()}\n'f'{Config.DATASET} | Cut Layer {Config.CUT_LAYER}',fontsize=12, fontweight='bold')
        save_path = f"{Config.RESULTS_DIR}/unsplit_{tag}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved → {save_path}")