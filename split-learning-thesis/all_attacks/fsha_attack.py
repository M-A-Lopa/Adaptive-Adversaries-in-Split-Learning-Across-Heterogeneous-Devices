import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from config import Config
from all_model.models import ClientModel
from all_model.ressfl_models import custom_AE, xavier_init


def compute_mse(original, reconstructed):
    return torch.mean((original - reconstructed) ** 2).item()


def compute_psnr(original, reconstructed):
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def _ssim_single(x, y):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y   = x.mean().item(), y.mean().item()
    sig_x, sig_y = x.var().item(), y.var().item()
    sig_xy       = ((x - mu_x) * (y - mu_y)).mean().item()
    numerator    = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
    denominator  = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x + sig_y + C2)
    return numerator / (denominator + 1e-8)


def compute_ssim(original, reconstructed):
    if original.dim() == 4:
        scores = [_ssim_single(original[i], reconstructed[i]) for i in range(original.shape[0])]
        return float(np.mean(scores))
    return _ssim_single(original, reconstructed)


def denormalize(tensor, dataset='CIFAR10'):
    if dataset == 'CIFAR10':
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(tensor.device)
        std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1).to(tensor.device)
    else:
        mean = torch.tensor([0.1307]).view(1, 1, 1).to(tensor.device)
        std  = torch.tensor([0.3081]).view(1, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)


class _AEWithBilinearFinal(nn.Module):
    def __init__(self, input_nc, output_nc, input_dim, intermediate_dim, target_dim, activation='sigmoid'):
        super().__init__()
        self.ae = custom_AE(input_nc=input_nc, output_nc=output_nc, input_dim=input_dim, output_dim=intermediate_dim, activation='relu')
        self.final = nn.Sequential(
            nn.Upsample(size=(target_dim, target_dim), mode='bilinear', align_corners=False),
            nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()
        )
        self.apply(xavier_init)

    def forward(self, x):
        return self.final(self.ae(x))


def _build_decoder_for_shape(smashed_shape, original_channels, original_spatial, activation='sigmoid'):
    input_nc, input_dim = smashed_shape[1], smashed_shape[2]
    ratio = original_spatial / input_dim
    log2_ratio = math.log2(ratio)

    if abs(log2_ratio - round(log2_ratio)) < 0.05:
        ae = custom_AE(input_nc=input_nc, output_nc=original_channels, input_dim=input_dim, output_dim=original_spatial, activation=activation)
        ae.apply(xavier_init)
        return ae
    else:
        upsample_steps = int(log2_ratio)
        intermediate_dim = input_dim * (2 ** upsample_steps)
        return _AEWithBilinearFinal(input_nc=input_nc, output_nc=original_channels, input_dim=input_dim,
                                     intermediate_dim=intermediate_dim, target_dim=original_spatial, activation=activation)


class SmashedDataCritic(nn.Module):

    def __init__(self, channels, spatial):
        super().__init__()
        c = max(channels, 8)
        self.net = nn.Sequential(
            nn.Conv2d(channels, c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(c * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((max(spatial // 2, 1), max(spatial // 2, 1))),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, spatial, spatial)
            flat_dim = self.net(dummy).shape[1]
        self.head = nn.Linear(flat_dim, 1)

    def forward(self, smashed):
        return self.head(self.net(smashed))


def gradient_penalty(critic, real, fake, device):
    batch_size = real.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=scores, inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


class FSHAAttack:

    def __init__(self, client_model, in_channels=3, dataset='CIFAR10',
                 pilot_builder=None, critic_iters=5, gp_lambda=10.0):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.in_channels = in_channels
        self.critic_iters = critic_iters
        self.gp_lambda = gp_lambda

        self.client_model = client_model.to(self.device)
        self.pilot = (pilot_builder() if pilot_builder is not None
                      else ClientModel(in_channels=in_channels)).to(self.device)

        img_size = 28 if dataset == 'MNIST' else 32
        smashed_shape = self._infer_smashed_shape(img_size)

        self.decoder = _build_decoder_for_shape(
            smashed_shape, original_channels=in_channels,
            original_spatial=img_size, activation='sigmoid'
        ).to(self.device)

        self.critic = SmashedDataCritic(smashed_shape[1], smashed_shape[2]).to(self.device)

        self.client_optimizer = optim.Adam(self.client_model.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.pilot_optimizer  = optim.Adam(list(self.pilot.parameters()) + list(self.decoder.parameters()), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

        self.mse = nn.MSELoss()
        self.history = {'critic_loss': [], 'hijack_loss': [], 'recon_loss': []}

        os.makedirs(Config.RESULTS_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("   FSHA (WGAN-GP) -- TRAINING-HIJACKING ATTACK")
        print("=" * 60)
        print("  Reference      : Pasquini et al., CCS 2021")
        print(f"  Dataset        : {dataset}")
        print(f"  Smashed shape  : {tuple(smashed_shape[1:])}")
        print(f"  Pilot params   : {sum(p.numel() for p in self.pilot.parameters()):,}")
        print(f"  Decoder params : {sum(p.numel() for p in self.decoder.parameters()):,}")
        print(f"  Critic iters   : {critic_iters} per round | GP lambda: {gp_lambda}")

    def _infer_smashed_shape(self, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, img_size, img_size, device=self.device)
            out = self.client_model.to(self.device)(dummy)
        return out.shape

    def hijack(self, private_loader, public_loader, epochs=5):
        print(f"\n  Running FSHA hijacking protocol for {epochs} epoch(s)...")

        self.client_model.train()
        self.pilot.train()
        self.decoder.train()
        self.critic.train()

        for epoch in range(epochs):
            pub_iter = iter(public_loader)
            running_c, running_h, running_r, n_batches = 0.0, 0.0, 0.0, 0

            progress = tqdm(private_loader, desc=f"  Hijack epoch [{epoch+1}/{epochs}]", leave=False)
            for priv_inputs, _ in progress:
                priv_inputs = priv_inputs.to(self.device)

                def next_pub():
                    nonlocal pub_iter
                    try:
                        x, _ = next(pub_iter)
                    except StopIteration:
                        pub_iter = iter(public_loader)
                        x, _ = next(pub_iter)
                    return x.to(self.device)

                pub_inputs = next_pub()
                self.pilot_optimizer.zero_grad()
                pilot_smashed = self.pilot(pub_inputs)
                recon = self.decoder(pilot_smashed)
                recon_loss = self.mse(recon, pub_inputs)
                recon_loss.backward()
                self.pilot_optimizer.step()

                with torch.no_grad():
                    client_smashed_fixed = self.client_model(priv_inputs)

                for _ in range(self.critic_iters):
                    pub_batch = next_pub()
                    with torch.no_grad():
                        real_smashed = self.pilot(pub_batch)
                    self.critic_optimizer.zero_grad()
                    d_real = self.critic(real_smashed)
                    d_fake = self.critic(client_smashed_fixed)
                    gp = gradient_penalty(self.critic, real_smashed, client_smashed_fixed, self.device)
                    critic_loss = d_fake.mean() - d_real.mean() + self.gp_lambda * gp
                    critic_loss.backward()
                    self.critic_optimizer.step()

                self.client_optimizer.zero_grad()
                client_smashed = self.client_model(priv_inputs)
                hijack_loss = -self.critic(client_smashed).mean()
                hijack_loss.backward()
                self.client_optimizer.step()

                running_c += critic_loss.item()
                running_h += hijack_loss.item()
                running_r += recon_loss.item()
                n_batches += 1
                progress.set_postfix({'Critic': f'{critic_loss.item():.3f}',
                                       'Hijack': f'{hijack_loss.item():.3f}',
                                       'Recon':  f'{recon_loss.item():.3f}'})

            self.history['critic_loss'].append(running_c / n_batches)
            self.history['hijack_loss'].append(running_h / n_batches)
            self.history['recon_loss'].append(running_r / n_batches)
            print(f"  Epoch {epoch+1:2d}/{epochs} | "
                  f"Critic: {running_c/n_batches:.4f} | "
                  f"Hijack: {running_h/n_batches:.4f} | "
                  f"Pilot-Recon: {running_r/n_batches:.4f}")

        print("\n  Hijacking complete.")

    def reconstruct(self, test_loader, num_images=32):
        self.client_model.eval()
        self.decoder.eval()

        all_psnr, all_ssim, all_mse = [], [], []
        originals_store, reconstructed_store = [], []
        n_seen = 0

        with torch.no_grad():
            for images, _ in test_loader:
                if n_seen >= num_images:
                    break
                images = images[:num_images - n_seen].to(self.device)

                smashed = self.client_model(images)
                reconstructed = self.decoder(smashed).clamp(0, 1)
                originals_dn = denormalize(images, self.dataset)

                for i in range(images.shape[0]):
                    orig, rec = originals_dn[i], reconstructed[i]
                    all_psnr.append(compute_psnr(orig, rec))
                    all_ssim.append(compute_ssim(orig.unsqueeze(0), rec.unsqueeze(0)))
                    all_mse.append(compute_mse(orig, rec))

                if n_seen == 0:
                    originals_store = originals_dn[:8].cpu()
                    reconstructed_store = reconstructed[:8].cpu()
                n_seen += images.shape[0]

        summary = {'mse': float(np.mean(all_mse)), 'psnr': float(np.mean(all_psnr)), 'ssim': float(np.mean(all_ssim))}

        print("\n" + "=" * 60)
        print("   FSHA RECONSTRUCTION RESULTS")
        print("=" * 60)
        print(f"  MSE  : {summary['mse']:.5f}")
        print(f"  PSNR : {summary['psnr']:.2f} dB")
        print(f"  SSIM : {summary['ssim']:.4f}")
        print("=" * 60)

        self._save_visualization(originals_store, reconstructed_store, tag='no_defense')
        return summary

    def _save_visualization(self, originals, reconstructed, tag='result'):
        num = min(8, len(originals))
        if num == 0:
            return
        fig = plt.figure(figsize=(num * 2, 5))
        gs = gridspec.GridSpec(2, num, hspace=0.3)
        for i in range(num):
            ax1 = fig.add_subplot(gs[0, i])
            img = originals[i].permute(1, 2, 0).numpy()
            ax1.imshow(img.squeeze(2), cmap='gray') if img.shape[2] == 1 else ax1.imshow(np.clip(img, 0, 1))
            ax1.axis('off')
            if i == 0:
                ax1.set_title('Original', fontsize=10, fontweight='bold')

            ax2 = fig.add_subplot(gs[1, i])
            rec = reconstructed[i].permute(1, 2, 0).numpy()
            ax2.imshow(rec.squeeze(2), cmap='gray') if rec.shape[2] == 1 else ax2.imshow(np.clip(rec, 0, 1))
            ax2.axis('off')
            if i == 0:
                ax2.set_title('Reconstructed\n(FSHA)', fontsize=10, fontweight='bold')

        plt.suptitle(f'FSHA -- {tag.replace("_", " ").title()}\n{self.dataset} | Cut Layer {Config.CUT_LAYER}',
                     fontsize=12, fontweight='bold')
        save_path = f"{Config.RESULTS_DIR}/fsha_{tag}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Visualization saved -> {save_path}")