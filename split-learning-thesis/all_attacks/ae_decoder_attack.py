# ae_decoder_attack.py
# Standalone AE Decoder Attack — Model-based Model Inversion
# Reusable against any defense: vanilla SL, PGSL, ResSFL
#
# Based on the MIA attack protocol from ResSFL (Li et al.)
# Mirrors attack() → test_attack() in MIA_torch.py
#
# preprocess_fn: optional callable applied to normalized images BEFORE
#   passing to client_model. Used for PGSL (space_to_depth_downsample).
#   Pass None for vanilla SL and ResSFL (images go directly to client).
#
# The AE decoder is built dynamically from the detected smashed data shape.
# Handles both power-of-2 and non-power-of-2 spatial ratios:
#   Vanilla CIFAR-10 smashed [B,32,8,8]  → output [B,3,32,32]  ratio=4 ✓
#   Vanilla MNIST   smashed [B,32,7,7]  → output [B,1,28,28]  ratio=4 ✓
#   PGSL    CIFAR-10 smashed [B,32,4,4]  → output [B,3,32,32]  ratio=8 ✓
#   PGSL    MNIST   smashed [B,32,3,3]  → output [B,1,28,28]  ratio≈9 handled with bilinear


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


# ── AE builder that handles non-power-of-2 spatial ratios ────────────────────
class _AEWithBilinearFinal(nn.Module):
    """
    Wraps custom_AE with a final bilinear upsample for cases where
    output_dim / input_dim is not a power of 2.

    Example: PGSL MNIST smashed 3×3 → original 28×28
      custom_AE: 3→6→12→24 (3 upsampling steps, nearest power-of-2 target)
      bilinear:  24→28 (exact target size)
    """

    def __init__(self, input_nc, output_nc, input_dim,
                 intermediate_dim, target_dim, activation='sigmoid'):
        super().__init__()
        # AE upsample to intermediate (power-of-2 aligned) size
        # Use ReLU output so bilinear can work on unclamped values
        self.ae = custom_AE(
            input_nc=input_nc, output_nc=output_nc,
            input_dim=input_dim, output_dim=intermediate_dim,
            activation='relu'
        )
        # Final bilinear resize to exact target + activation clamp
        self.final = nn.Sequential(
            nn.Upsample(size=(target_dim, target_dim),
                        mode='bilinear', align_corners=False),
            nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()
        )
        self.apply(xavier_init)

    def forward(self, x):
        return self.final(self.ae(x))


def _build_ae_for_shape(smashed_shape, original_channels,
                         original_spatial, activation='sigmoid'):
    """
    Builds a decoder AE matched to the actual smashed data shape.

    smashed_shape:    (B, C, H, W) — from a real forward pass
    original_channels: 3 for CIFAR-10, 1 for MNIST
    original_spatial:  32 for CIFAR-10, 28 for MNIST
    """
    input_nc  = smashed_shape[1]   # channels: 32
    input_dim = smashed_shape[2]   # spatial:  8, 7, 4, or 3

    ratio     = original_spatial / input_dim
    log2_ratio = math.log2(ratio)

    if abs(log2_ratio - round(log2_ratio)) < 0.05:
        # Exact power-of-2 ratio — custom_AE handles it directly
        ae = custom_AE(
            input_nc=input_nc, output_nc=original_channels,
            input_dim=input_dim, output_dim=original_spatial,
            activation=activation
        )
        ae.apply(xavier_init)
        return ae
    else:
        # Non-power-of-2: compute intermediate dim, add bilinear correction
        upsample_steps  = int(log2_ratio)              # floor
        intermediate_dim = input_dim * (2 ** upsample_steps)
        return _AEWithBilinearFinal(
            input_nc=input_nc,
            output_nc=original_channels,
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            target_dim=original_spatial,
            activation=activation
        )


def _detect_smashed_shape(client_model, data_loader, preprocess_fn, device):
    """Runs one batch to detect smashed data spatial shape."""
    client_model.eval()
    for inputs, _ in data_loader:
        inputs = inputs[:4].to(device)
        if preprocess_fn is not None:
            with torch.no_grad():
                inputs = preprocess_fn(inputs)
        with torch.no_grad():
            smashed = client_model(inputs)
        return smashed.shape   # [B, C, H, W]
    raise ValueError("Empty data loader — cannot detect smashed shape.")


# ── Main attack function ──────────────────────────────────────────────────────
def run_ae_decoder_attack(client_model, train_loader, test_loader,
                           device, dataset,
                           preprocess_fn=None,
                           ae_epochs=50,
                           ae_batch_size=32,
                           collect_train_batches=80,
                           collect_test_batches=32,
                           label='AE Decoder'):
    """
    Trains a fresh AE decoder to reconstruct original images from
    smashed data produced by the defended client model.
    This is the original ResSFL paper's evaluation protocol.

    Parameters:
    -----------
    client_model: the defended client model (vanilla, PGSL, or ResSFL)
    train_loader: used to collect (image, smashed) training pairs
    test_loader:  used to collect evaluation pairs
    device:       cuda or cpu
    dataset:      'CIFAR10' or 'MNIST' — for denormalization and AE sizing
    preprocess_fn: callable(inputs) → processed_inputs applied BEFORE client
                   Use PGSLDefenseModules.space_to_depth_downsample for PGSL
                   Pass None for vanilla SL and ResSFL
    ae_epochs:    training epochs for the fresh decoder (50 as in original)
    label:        string label for printed output and visualization title

    Returns:
    --------
    summary:          {'mean_mse', 'mean_ssim', 'mean_psnr'}
    originals_vis:    first 8 original images (tensor, cpu)
    reconstructed_vis: first 8 reconstructed images (tensor, cpu)
    """
    print("\n" + "=" * 60)
    print(f"  AE DECODER ATTACK — {label}")
    print("=" * 60)
    print(f"  Training fresh AE for {ae_epochs} epochs")
    print(f"  Dataset : {dataset}")

    client_model.eval()

    # ── Determine output image dimensions ────────────────────────────────────
    original_channels = 1 if dataset == 'MNIST' else 3
    original_spatial  = 28 if dataset == 'MNIST' else 32

    # ── Detect smashed data shape from live forward pass ─────────────────────
    smashed_shape = _detect_smashed_shape(
        client_model, train_loader, preprocess_fn, device
    )
    print(f"  Smashed data shape : {list(smashed_shape)}")

    # ── Step 1: Collect (denorm_image, smashed) training pairs ───────────────
    all_imgs    = []
    all_smashed = []

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx >= collect_train_batches:
                break
            inputs = inputs.to(device)

            # Apply preprocessing if needed (e.g. PGSL space_to_depth)
            if preprocess_fn is not None:
                processed = preprocess_fn(inputs)
            else:
                processed = inputs

            smashed = client_model(processed)

            # Original image denormalized to [0,1] for reconstruction target
            imgs_dn = denormalize(inputs, dataset)

            all_imgs.append(imgs_dn.cpu())
            all_smashed.append(smashed.cpu())

    all_imgs    = torch.cat(all_imgs,    dim=0)
    all_smashed = torch.cat(all_smashed, dim=0)
    print(f"  Collected {len(all_imgs)} training pairs")

    # ── Step 2: Train / val split (90/10 as in original) ─────────────────────
    n_train  = int(0.9 * len(all_imgs))
    ae_train = DataLoader(
        TensorDataset(all_imgs[:n_train], all_smashed[:n_train]),
        batch_size=ae_batch_size, shuffle=True
    )
    ae_val = DataLoader(
        TensorDataset(all_imgs[n_train:], all_smashed[n_train:]),
        batch_size=ae_batch_size, shuffle=False
    )

    # ── Step 3: Build fresh decoder (no knowledge of defense internals) ──────
    attack_ae        = _build_ae_for_shape(
        smashed_shape, original_channels, original_spatial, activation='sigmoid'
    ).to(device)
    ae_optimizer     = torch.optim.Adam(attack_ae.parameters(), lr=1e-3)
    mse_criterion    = nn.MSELoss()
    ssim_criterion   = WindowedSSIM().to(device)

    print(f"  AE decoder params  : "
          f"{sum(p.numel() for p in attack_ae.parameters()):,}")

    # ── Step 4: Train the decoder ─────────────────────────────────────────────
    # Best checkpoint saved by minimum validation MSE (mirrors original)
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

        # Save best decoder state
        if val_mse < best_val_mse:
            best_val_mse  = val_mse
            best_ae_state = {k: v.clone() for k, v in
                             attack_ae.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{ae_epochs} | "
                  f"Val MSE: {val_mse:.5f}")

    # Load best checkpoint before evaluation
    if best_ae_state is not None:
        attack_ae.load_state_dict(best_ae_state)
    attack_ae.eval()

    print(f"  Training complete. Best Val MSE: {best_val_mse:.5f}")

    # ── Step 5: Collect evaluation pairs from test set ────────────────────────
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

    test_dset = DataLoader(
        TensorDataset(test_imgs, test_smashed),
        batch_size=ae_batch_size, shuffle=False
    )

    # ── Step 6: Evaluate reconstruction quality ───────────────────────────────
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

            # Save first batch for visualization
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

    summary = {
        'mean_mse'  : float(np.mean(all_mse)),
        'mean_ssim' : float(np.mean(all_ssim)),
        'mean_psnr' : float(np.mean(all_psnr))
    }

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


# ── Visualization helper (usable by both pgsl_run and ressfl_run) ─────────────
def save_ae_attack_visualization(originals, reconstructed,
                                  baseline_summary, defense_summary,
                                  defense_name, dataset, results_dir):
    """
    Side-by-side: original (top) vs AE-reconstructed (bottom).
    Saves to results_dir with a filename including defense_name.
    """
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
        fontsize=10, fontweight='bold'
    )

    tag  = defense_name.lower().replace(' ', '_')
    path = f"{results_dir}/ae_decoder_{tag}_{dataset}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved → {path}")