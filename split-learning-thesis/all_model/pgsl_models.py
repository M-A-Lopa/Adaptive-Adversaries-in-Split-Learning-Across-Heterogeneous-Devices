# pgsl_models.py
# PGSL-specific model architectures
#
# PGSLClientModel:
#   Same conv1+conv2 structure as ClientModel but accepts (4*C + 1) input channels.
#   For MNIST (1 channel):   input is [B, 5,  H/2, W/2]
#   For CIFAR-10 (3 channels): input is [B, 13, H/2, W/2]
#
# PGSLServerModel:
#   Three separate conv3+fc streams (attacked, recovered, fused).
#   Plus ProximalRecoveryBlock and ConvolutionSumFusion.
#   Maintains 3-conv + 1-fc architecture per stream.
#
# Architecture per stream (server side):
#   smashed [B,32,H,W] → Conv3(32→64,3×3) → BN → ReLU → AdaptiveAvgPool(4,4)
#                       → Flatten → Linear(1024→256) → ReLU → Dropout
#                       → Linear(256→num_classes)

import torch
import torch.nn as nn
from all_defences.pgsl_defense import (PGSLProximalRecoveryBlock, ConvolutionSumFusion)


class PGSLClientModel(nn.Module):
    """
    Client-side model for PGSL split learning.
    Identical conv structure to ClientModel but accepts 4*C+1 input channels
    because input has been processed through space_to_depth_downsample first.

    Input channels:
    - MNIST   original_channels=1: pgsl_channels = 4*1+1 = 5
    - CIFAR-10 original_channels=3: pgsl_channels = 4*3+1 = 13
    """

    def __init__(self, original_in_channels=1):
        super(PGSLClientModel, self).__init__()

        # 4*C + 1 input channels due to space_to_depth + saliency map
        pgsl_in_channels = 4 * original_in_channels + 1

        # Conv1: same structure as ClientModel, different input channels
        # MNIST:   [B, 5,  14, 14] → [B, 16, 7, 7]
        # CIFAR-10: [B, 13, 16, 16] → [B, 16, 8, 8]
        self.conv1 = nn.Sequential(
            nn.Conv2d(pgsl_in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv2: cut layer — output is smashed data
        # MNIST:   [B, 16, 7, 7] → [B, 32, 3, 3]  (7//2=3)
        # CIFAR-10: [B, 16, 8, 8] → [B, 32, 4, 4]
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # Smashed data


class PGSLServerModel(nn.Module):
    """
    Server-side PGSL model with three independent streams.

    Stream A (attacked):  raw smashed data → conv3_a → classifier_a
    Stream R (recovered): recovery(smashed) → conv3_r → classifier_r
    Stream F (fused):     fusion(smashed, recovered) → conv3_f → classifier_f

    run_full_pipeline=False: only runs stream A (for JSMA baseline logits)
    run_full_pipeline=True:  runs all three streams (for training/evaluation)

    Architecture per stream maintains 3-conv+1-fc structure:
    [conv3 server-side] + [adaptive pool] + [fc classifier]
    """

    def __init__(self, num_classes=10, smashed_channels=32):
        super(PGSLServerModel, self).__init__()

        self.recovery = PGSLProximalRecoveryBlock(mu=0.55)
        self.fusion   = ConvolutionSumFusion(smashed_channels)

        # Three independent server-side conv3 backbones
        self.stream_a = self._make_backbone(smashed_channels)
        self.stream_r = self._make_backbone(smashed_channels)
        self.stream_f = self._make_backbone(smashed_channels)

        # Three independent classifiers
        self.classifier_a = self._make_classifier(num_classes)
        self.classifier_r = self._make_classifier(num_classes)
        self.classifier_f = self._make_classifier(num_classes)

    def _make_backbone(self, in_channels):
        """
        Conv3 + AdaptiveAvgPool.
        AdaptiveAvgPool maps any spatial size to fixed 4×4.
        This handles variable smashed data dimensions across datasets.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def _make_classifier(self, num_classes):
        """FC classification head. 64*4*4=1024 input features."""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, smashed_data, run_full_pipeline=True):
        """
        smashed_data: [B, 32, H, W] received from PGSLClientModel

        run_full_pipeline=False: stream A only (for JSMA Jacobian computation)
        run_full_pipeline=True:  all three streams
        """
        # ── Stream A: attacked (raw smashed data) ─────────────────────
        features_a = self.stream_a(smashed_data)
        out_a      = self.classifier_a(features_a)

        if not run_full_pipeline:
            return out_a, None, None

        # ── Stream R: recovered (proximal gradient applied) ───────────
        smashed_r  = self.recovery(smashed_data)
        features_r = self.stream_r(smashed_r)
        out_r      = self.classifier_r(features_r)

        # ── Stream F: fused (conv-sum fusion of A and R) ──────────────
        smashed_f  = self.fusion(smashed_data, smashed_r)
        features_f = self.stream_f(smashed_f)
        out_f      = self.classifier_f(features_f)

        return out_a, out_r, out_f