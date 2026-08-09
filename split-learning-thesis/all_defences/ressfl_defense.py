# ressfl_defense.py
# ResSFL Defense Utilities
# Based on: ResSFL — A Resistance Transfer Framework for Defending
# Model Inversion Attack in Split Federated Learning
#
# Contains:
# 1. WindowedSSIM — exact port of pytorch_ssim.py from ResSFL repo
#    Uses 11×11 Gaussian window, per-channel local statistics.
#    This is the loss used in both AE training and client adversarial training.
# 2. denormalize() — reverses dataset normalization to [0,1] for SSIM computation.
#    Must match the normalization applied in dataset.py exactly.
# 3. xavier_init() — weight initialization from ResSFL repo.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np


# ── Gaussian window for SSIM ──────────────────────────────────────────────────
def _gaussian(window_size, sigma):
    """
    1D Gaussian kernel used to build the 2D SSIM window.
    window_size=11, sigma=1.5 as in the original ResSFL pytorch_ssim.
    """
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    """
    Creates a 2D Gaussian window by outer product of two 1D Gaussians.
    Expanded to [channel, 1, window_size, window_size] for grouped conv.
    """
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim_compute(img1, img2, window, window_size, channel, size_average=True):
    """
    Core SSIM computation using local Gaussian-weighted statistics.
    Exact port from pytorch_ssim.py in ResSFL repository.

    Local means via depthwise conv2d (groups=channel).
    Local variances and covariance via E[X²] - (E[X])².

    C1 = 0.01² = 1e-4
    C2 = 0.03² = 9e-4

    ssim_map = ((2·μ1·μ2 + C1)(2·σ12 + C2)) /
               ((μ1² + μ2² + C1)(σ1² + σ2² + C2))
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2,
                         groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2,
                         groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12   + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class WindowedSSIM(nn.Module):
    """
    Windowed SSIM loss module — direct port of pytorch_ssim.SSIM from ResSFL.

    window_size=11, sigma=1.5 (standard Wang et al. 2004 parameters).
    Window is cached and reused if channel count does not change.

    Used in two places:
    1. gan_train_step: loss = -SSIM(AE(z), x_denorm)   [maximize SSIM]
    2. train_target_step: loss += α * SSIM(AE(z), x_denorm) [minimize SSIM]

    Both operate on images in [0, 1] range (after denormalization).
    Using normalized tensors would give meaningless SSIM values.
    """

    def __init__(self, window_size=11, size_average=True):
        super(WindowedSSIM, self).__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.channel      = 1
        self.window       = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """
        img1, img2: [B, C, H, W] in range [0, 1].
        Returns scalar SSIM value in [-1, 1], typically [0, 1] for natural images.
        """
        (_, channel, _, _) = img1.size()

        # Recreate window if channel count changed (e.g. MNIST→CIFAR-10 switch)
        if channel == self.channel and \
                self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window  = window
            self.channel = channel

        return _ssim_compute(img1, img2, window,
                             self.window_size, channel, self.size_average)


# ── Denormalization ───────────────────────────────────────────────────────────
def denormalize(x, dataset):
    """
    Reverses dataset normalization, mapping tensors back to [0, 1].
    Must use the SAME mean and std as applied in DatasetLoader (dataset.py).

    Original ResSFL uses slightly different CIFAR-10 std but we match our
    existing pipeline to ensure consistency across all experiments.

    x: [B, C, H, W] normalized tensor
    dataset: 'CIFAR10' or 'MNIST'
    Returns: [B, C, H, W] in [0, 1]
    """
    if dataset == 'CIFAR10':
        # Must match dataset.py: Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
        mean = torch.tensor([0.4914, 0.4822, 0.4465],
                            device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010],
                            device=x.device).view(1, 3, 1, 1)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    elif dataset == 'MNIST':
        # Must match dataset.py: Normalize((0.1307,),(0.3081,))
        mean = torch.tensor([0.1307], device=x.device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081], device=x.device).view(1, 1, 1, 1)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'CIFAR10' or 'MNIST'.")


# ── Weight initialization ─────────────────────────────────────────────────────
def xavier_init(m):
    """
    Xavier uniform initialization for Linear and Conv layers.
    Direct port from init_weights() in MIA_torch.py.
    Applied to AE decoder at initialization.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()