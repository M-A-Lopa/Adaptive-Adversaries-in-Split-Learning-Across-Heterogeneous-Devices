import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd


class AttackMetricsTracker:
    def __init__(self):
        self.psnr_values = []
        self.ssim_values = []

    @staticmethod
    def calculate_psnr(img1, img2):
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            return float('inf')
        max_i = 1.0
        return 10 * np.log10((max_i ** 2) / mse)

    @staticmethod
    def calculate_ssim(img1, img2):
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        img1_m    = img1 - mu1
        img2_m    = img2 - mu2
        covariance = torch.mean(img1_m * img2_m)

        c1 = (0.01 ** 2)
        c2 = (0.03 ** 2)

        numerator   = (2 * mu1 * mu2 + c1) * (2 * covariance + c2)
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)

        return torch.clamp(numerator / denominator, 0, 1).item()

    def log_batch(self, original_batch, reconstructed_batch):
        for orig, recon in zip(original_batch, reconstructed_batch):
            self.psnr_values.append(self.calculate_psnr(orig, recon))
            self.ssim_values.append(self.calculate_ssim(orig, recon))

    def get_summary(self):
        return {
            "mean_psnr": np.mean(self.psnr_values),
            "mean_ssim": np.mean(self.ssim_values)
        }


class WhiteBoxInversionAttack:
    def __init__(self, client_model, dataset,lambda_tv=1e-4, iterations=2000, lr=1e-2, use_normalization=True):
        self.client_model = client_model
        self.client_model.eval()
        self.lambda_tv  = lambda_tv
        self.iterations = iterations
        self.lr         = lr
        self.dataset    = dataset
        self.use_normalization = use_normalization

        # Normalization parameters — must match DatasetLoader transforms
        if dataset == 'CIFAR10':
            self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
            self.std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
        else:  # MNIST
            self.mean = torch.tensor([0.1307]).view(1, 1, 1, 1)
            self.std  = torch.tensor([0.3081]).view(1, 1, 1, 1)

    def _total_variation_loss(self, x):
        # Squared differences in both spatial directions
        diff_h = (x[:, :, 1:, :] - x[:, :, :-1, :]) ** 2
        diff_w = (x[:, :, :, 1:] - x[:, :, :, :-1]) ** 2

        # Pad to match dimensions before summing
        # diff_h: [B, C, H-1, W]  diff_w: [B, C, H, W-1]
        # Take minimum spatial extent for element-wise combination
        h_min = min(diff_h.shape[2], diff_w.shape[2])
        w_min = min(diff_h.shape[3], diff_w.shape[3])

        # Isotropic: square root of sum of squared spatial differences
        iso_tv = torch.sqrt(
            diff_h[:, :, :h_min, :w_min] +
            diff_w[:, :, :h_min, :w_min] +
            1e-8  # numerical stability inside sqrt
        )
        return torch.mean(iso_tv)

    def _reconstruct_single(self, target_smashed_single, single_input_shape):
        device = target_smashed_single.device

        mean = self.mean.to(device)
        std  = self.std.to(device)

        reconstructed_x = torch.full(
            single_input_shape, 0.5,
            device=device, requires_grad=True
        )

        optimizer = optim.Adam([reconstructed_x], lr=self.lr)
        criterion = nn.MSELoss()

        for step in range(self.iterations):
            optimizer.zero_grad()
            if self.use_normalization:
                normalized_x = (reconstructed_x - mean) / std
            else:
                normalized_x = reconstructed_x
            current_smashed = self.client_model(normalized_x)
            loss_ed = criterion(current_smashed, target_smashed_single)

            loss_tv = self._total_variation_loss(reconstructed_x)

            total_loss = loss_ed + self.lambda_tv * loss_tv
            total_loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0 or step == 0:
                print(
                    f"Iteration {step+1}/{self.iterations} | "
                    f"Feature Loss: {loss_ed.item():.6f} | "
                    f"TV Loss: {loss_tv.item():.6f} | "
                    f"Total Loss: {total_loss.item():.6f}"
                )

            with torch.no_grad():
                reconstructed_x.clamp_(0.0, 1.0)

        return reconstructed_x.detach()

    def reconstruct(self, target_smashed_data, input_shape):
        batch_size     = target_smashed_data.shape[0]
        single_shape   = (1, input_shape[1], input_shape[2], input_shape[3])
        reconstructed_list = []

        for i in range(batch_size):
            single_smashed = target_smashed_data[i:i+1]  
            single_recon   = self._reconstruct_single(single_smashed, single_shape)
            reconstructed_list.append(single_recon)

        return torch.cat(reconstructed_list, dim=0)