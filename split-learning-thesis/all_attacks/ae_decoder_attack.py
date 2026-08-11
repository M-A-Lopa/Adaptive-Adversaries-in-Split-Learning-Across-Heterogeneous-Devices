import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import TensorDataset, DataLoader

from all_model.ressfl_models import custom_AE, xavier_init
from all_defences.ressfl_defense import WindowedSSIM, denormalize
from config import Config


class _AEWithBilinearFinal(nn.Module):

    def __init__(self, input_nc, output_nc, input_dim, intermediate_dim, target_dim, activation='sigmoid'):
        super().__init__()
        self.ae = custom_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=intermediate_dim, activation='relu')
        self.final = nn.Sequential(
            nn.Upsample(size=(target_dim, target_dim),
                        mode='bilinear', align_corners=False),
            nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()
        )
        self.apply(xavier_init)

    def forward(self, x):
        return self.final(self.ae(x))


def _build_ae_for_shape(smashed_shape, original_channels, original_spatial, activation='sigmoid'):

    input_nc  = smashed_shape[1]   
    input_dim = smashed_shape[2]  

    ratio     = original_spatial / input_dim
    log2_ratio = math.log2(ratio)

    if abs(log2_ratio - round(log2_ratio)) < 0.05:

        ae = custom_AE(input_nc=input_nc, output_nc=original_channels, input_dim=input_dim, output_dim=original_spatial, activation=activation)
        ae.apply(xavier_init)
        return ae
    
    else:

        upsample_steps  = int(log2_ratio)            
        intermediate_dim = input_dim * (2 ** upsample_steps)
        return _AEWithBilinearFinal( input_nc=input_nc, output_nc=original_channels, input_dim=input_dim, intermediate_dim=intermediate_dim, target_dim=original_spatial, activation=activation)


def _detect_smashed_shape(client_model, data_loader, preprocess_fn, device):

    client_model.eval()
    for inputs, _ in data_loader:
        inputs = inputs[:4].to(device)
        if preprocess_fn is not None:
            with torch.no_grad():
                inputs = preprocess_fn(inputs)
        with torch.no_grad():
            smashed = client_model(inputs)
        return smashed.shape  
    raise ValueError("Empty data loader — cannot detect smashed shape.")



def run_ae_decoder_attack(client_model, train_loader, test_loader, device, dataset, preprocess_fn=None, ae_epochs=50, ae_batch_size=32, collect_train_batches=80, collect_test_batches=32, label='AE Decoder'):

    print("\n" + "=" * 60)
    print(f"  AE DECODER ATTACK — {label}")
    print("=" * 60)
    print(f"  Training fresh AE for {ae_epochs} epochs")
    print(f"  Dataset : {dataset}")

    client_model.eval()

    original_channels = 1 if dataset == 'MNIST' else 3
    original_spatial  = 28 if dataset == 'MNIST' else 32

    smashed_shape = _detect_smashed_shape(client_model, train_loader, preprocess_fn, device)
    print(f"  Smashed data shape : {list(smashed_shape)}")

    all_imgs    = []
    all_smashed = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx >= collect_train_batches:
                break
            inputs = inputs.to(device)

            if preprocess_fn is not None:
                processed = preprocess_fn(inputs)
            else:
                processed = inputs

            smashed = client_model(processed)

            imgs_dn = denormalize(inputs, dataset)

            all_imgs.append(imgs_dn.cpu())
            all_smashed.append(smashed.cpu())

    all_imgs    = torch.cat(all_imgs,    dim=0)
    all_smashed = torch.cat(all_smashed, dim=0)
    print(f"  Collected {len(all_imgs)} training pairs")

    n_train  = int(0.9 * len(all_imgs))
    ae_train = DataLoader(TensorDataset(all_imgs[:n_train], all_smashed[:n_train]), batch_size=ae_batch_size, shuffle=True)
    ae_val = DataLoader(TensorDataset(all_imgs[n_train:], all_smashed[n_train:]), batch_size=ae_batch_size, shuffle=False)

    attack_ae        = _build_ae_for_shape(smashed_shape, original_channels, original_spatial, activation='sigmoid').to(device)
    ae_optimizer     = torch.optim.Adam(attack_ae.parameters(), lr=1e-3)
    mse_criterion    = nn.MSELoss()
    ssim_criterion   = WindowedSSIM().to(device)

    print(f"  AE decoder params  : "
          f"{sum(p.numel() for p in attack_ae.parameters()):,}")

    best_val_mse  = float('inf')
    best_ae_state = None

    for epoch in range(ae_epochs):
        attack_ae.train()
        for imgs, smashed in ae_train:
            imgs    = imgs.to(device)
            smashed = smashed.to(device)
            ae_optimizer.zero_grad()
            recon = attack_ae(smashed)
            loss  = mse_criterion(recon, imgs)
            loss.backward()
            ae_optimizer.step()

        attack_ae.eval()
        val_mse = 0.0
        with torch.no_grad():
            for imgs, smashed in ae_val:
                imgs    = imgs.to(device)
                smashed = smashed.to(device)
                recon   = attack_ae(smashed)
                val_mse += mse_criterion(recon, imgs).item()
        val_mse /= max(len(ae_val), 1)

        if val_mse < best_val_mse:
            best_val_mse  = val_mse
            best_ae_state = {k: v.clone() for k, v in attack_ae.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{ae_epochs} | "
                  f"Val MSE: {val_mse:.5f}")

    if best_ae_state is not None:
        attack_ae.load_state_dict(best_ae_state)
    attack_ae.eval()

    print(f"  Training complete. Best Val MSE: {best_val_mse:.5f}")

    test_imgs    = []
    test_smashed = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            if batch_idx >= collect_test_batches:
                break
            inputs = inputs.to(device)

            if preprocess_fn is not None:
                processed = preprocess_fn(inputs)
            else:
                processed = inputs

            smashed = client_model(processed)
            imgs_dn = denormalize(inputs, dataset)

            test_imgs.append(imgs_dn.cpu())
            test_smashed.append(smashed.cpu())

    test_imgs    = torch.cat(test_imgs,    dim=0)
    test_smashed = torch.cat(test_smashed, dim=0)

    test_dset = DataLoader(TensorDataset(test_imgs, test_smashed), batch_size=ae_batch_size, shuffle=False)

    all_mse  = []
    all_ssim = []
    all_psnr = []

    originals_vis     = []
    reconstructed_vis = []
    first_batch_saved = False

    with torch.no_grad():
        for imgs, smashed in test_dset:
            imgs    = imgs.to(device)
            smashed = smashed.to(device)
            recon   = attack_ae(smashed)

            if not first_batch_saved:
                originals_vis     = imgs[:8].cpu()
                reconstructed_vis = recon[:8].cpu()
                first_batch_saved = True

            for i in range(imgs.shape[0]):
                mse_val  = torch.mean((imgs[i] - recon[i]) ** 2).item()
                ssim_val = ssim_criterion(
                    imgs[i].unsqueeze(0), recon[i].unsqueeze(0)
                ).item()
                psnr_val = 10 * np.log10(1.0 / (mse_val + 1e-10))
                all_mse.append(mse_val)
                all_ssim.append(ssim_val)
                all_psnr.append(psnr_val)

    summary = {'mean_mse'  : float(np.mean(all_mse)), 'mean_ssim' : float(np.mean(all_ssim)), 'mean_psnr' : float(np.mean(all_psnr))}

    print("\n" + "=" * 60)
    print(f"  {label} RESULTS")
    print("=" * 60)
    print(f"  MSE  : {summary['mean_mse']:.5f}")
    print(f"  PSNR : {summary['mean_psnr']:.2f} dB")
    print(f"  SSIM : {summary['mean_ssim']:.4f}")
    tag = "✓ below 0.3 — attack defeated" if summary['mean_ssim'] < 0.3 \
          else "above 0.3 — partial reconstruction"
    print(f"  Note : {tag}")
    print("=" * 60)

    return summary, originals_vis, reconstructed_vis


def save_ae_attack_visualization(originals, reconstructed, baseline_summary, defense_summary, defense_name, dataset, results_dir):

    num = min(8, len(originals))
    fig = plt.figure(figsize=(num * 2, 5))
    gs  = gridspec.GridSpec(2, num, hspace=0.35)

    for i in range(num):
        ax1 = fig.add_subplot(gs[0, i])
        img = originals[i].permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            ax1.imshow(img.squeeze(2), cmap='gray')
        else:
            ax1.imshow(np.clip(img, 0, 1))
        ax1.axis('off')
        if i == 0:
            ax1.set_title('Original', fontsize=10, fontweight='bold')

        ax2 = fig.add_subplot(gs[1, i])
        rec = reconstructed[i].permute(1, 2, 0).numpy()
        if rec.shape[2] == 1:
            ax2.imshow(rec.squeeze(2), cmap='gray')
        else:
            ax2.imshow(np.clip(rec, 0, 1))
        ax2.axis('off')
        if i == 0:
            ax2.set_title('Reconstructed\n(AE Decoder)',
                          fontsize=10, fontweight='bold')

    plt.suptitle(
        f'AE Decoder Attack vs {defense_name} — {dataset}\n'
        f'Baseline: PSNR={baseline_summary["mean_psnr"]:.2f}dB '
        f'SSIM={baseline_summary["mean_ssim"]:.4f} | '
        f'{defense_name}: PSNR={defense_summary["mean_psnr"]:.2f}dB '
        f'SSIM={defense_summary["mean_ssim"]:.4f}',
        fontsize=10, fontweight='bold' )

    tag  = defense_name.lower().replace(' ', '_')
    path = f"{results_dir}/ae_decoder_{tag}_{dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved → {path}")