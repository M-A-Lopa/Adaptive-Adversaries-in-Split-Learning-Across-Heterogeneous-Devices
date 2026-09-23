import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from config import Config

def total_variation(x):
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
    count_h = x[:, :, 1:, :].numel()
    count_w = x[:, :, :, 1:].numel()
    return (h_tv / count_h + w_tv / count_w) / x.size(0)

def l2_loss(x):
    return (x ** 2).mean()

def denormalize(tensor, dataset='CIFAR10'):

    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
        
    else: 
        mean = torch.tensor([0.1307]).view(1, 1, 1).to(tensor.device)
        std  = torch.tensor([0.3081]).view(1, 1, 1).to(tensor.device)

    return torch.clamp(tensor * std + mean, 0.0, 1.0)


def normalize_for_client(x, dataset='CIFAR10'):
    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(x.device)
    else:
        mean = torch.tensor([0.1307]).view(1, 1, 1, 1).to(x.device)
        std  = torch.tensor([0.3081]).view(1, 1, 1, 1).to(x.device)
    return (x - mean) / std


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
    print("  Higher MSE, lower PSNR, lower SSIM = stronger defense (harder to reconstruct)")
    print("  Higher Accuracy      = better utility preservation")
    
class UnSplitAttack:

    def __init__(self, client_model, in_channels=3, clone_builder=None,
                 main_iters=100, input_iters=20, model_iters=20):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')

        self.client_model = client_model.to(self.device)
        self.client_model.eval()

        if clone_builder is None:
            raise ValueError("clone_builder must be provided for UnSplit attack.")
        self.clone_builder = clone_builder

        self.mse_loss = nn.MSELoss()

        self.main_iters  = main_iters
        self.input_iters = input_iters
        self.model_iters = model_iters

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

        probe_clone = self.clone_builder()
        print("\n" + "="*60)
        print("   UNSPLIT ATTACK — INITIALISED")
        print("="*60)
        print(f"  Attack type    : Data-Oblivious Model Inversion")
        print(f"  Requires data  : NO (architecture knowledge only)")
        print(f"  Iters (main/input/model) : {self.main_iters}/{self.input_iters}/{self.model_iters}")
        print(f"  Clone params   : "
              f"{sum(p.numel() for p in probe_clone.parameters()):,}")
        del probe_clone

    def _reconstruct_batch(self, clone_model, smashed_data, input_shape,
                            lambda_tv=0.1, lambda_l2=1.0,
                            lr_input=0.001, lr_model=0.001):

        batch_size = smashed_data.shape[0]
        target = smashed_data.detach()

        x_pred = torch.full((batch_size, *input_shape), 0.5,
                             device=self.device, requires_grad=True)

        input_optimizer = optim.Adam([x_pred], lr=lr_input, amsgrad=True)
        model_optimizer = optim.Adam(clone_model.parameters(), lr=lr_model, amsgrad=True)

        for _ in range(self.main_iters):

            clone_model.eval()
            for _ in range(self.input_iters):
                input_optimizer.zero_grad()
                pred = clone_model(normalize_for_client(x_pred, Config.DATASET))
                loss = (self.mse_loss(pred, target)
                        + lambda_tv * total_variation(x_pred)
                        + lambda_l2 * l2_loss(x_pred))
                loss.backward()
                input_optimizer.step()
                with torch.no_grad():
                    x_pred.clamp_(0.0, 1.0)

            clone_model.train()
            for _ in range(self.model_iters):
                model_optimizer.zero_grad()
                pred = clone_model(normalize_for_client(x_pred.detach(), Config.DATASET))
                loss = self.mse_loss(pred, target)
                loss.backward()
                model_optimizer.step()

        clone_model.eval()
        return x_pred.detach()

    def run_attack(self, data_loader, num_batches=20):

        print(f"\n  Running UnSplit attack on {num_batches} batches...")
        print(f"  Main/input/model iters: {self.main_iters}/{self.input_iters}/{self.model_iters}")

        if Config.DATASET == 'MNIST':
            input_shape = (1, 28, 28)
        else:
            input_shape = (3, 32, 32)

        clone_model = self.clone_builder().to(self.device)

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

            reconstructed = self._reconstruct_batch(clone_model, smashed_data, input_shape)

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
        print("  After defense: MSE should increase, while PSNR and SSIM should decrease.")

        self._save_visualization(originals_store, reconstructed_store, tag='no_defense')

        return {'mse' : mean_mse, 'psnr': mean_psnr, 'ssim': mean_ssim}

    def run_attack_with_defense(self, data_loader, defense_fn, defense_name, num_batches=20):

        print(f"\n  Running UnSplit attack WITH defense: {defense_name}")

        if Config.DATASET == 'MNIST':
            input_shape = (1, 28, 28)
        else:
            input_shape = (3, 32, 32)

        clone_model = self.clone_builder().to(self.device)

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

            reconstructed = self._reconstruct_batch(clone_model, smashed_protected, input_shape)

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
        save_path = f"{Config.RESULTS_DIR}/{Config.MODEL_NAME.lower()}_unsplit_{tag}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved → {save_path}")