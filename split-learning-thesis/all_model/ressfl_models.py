# ressfl_models.py
# ResSFL Model Architectures
# Based on architectures_torch.py from ResSFL repository
#
# ResBlock: Pre-activation residual block used in res_normN_AE.
#           No ReLU after addition — matches original exactly.
#
# custom_AE: Primary attacker decoder used in ResSFL.
#   Dynamically computes number of upsampling steps from
#   input_dim and output_dim. No BatchNorm (unlike custom_AE_bn).
#   Activation: sigmoid (output in [0,1] matches denormalized images).
#
# Smashed data shapes from our 3-conv shallow ClientModel:
#   CIFAR-10: [B, 32, 8, 8]  → input_nc=32, input_dim=8, output_dim=32
#   MNIST:    [B, 32, 7, 7]  → input_nc=32, input_dim=7, output_dim=28
#             Note: log2(28/7)=2 so upsampling_num=2, same as CIFAR-10.


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from all_defences.ressfl_defense import xavier_init


# ── ResBlock ──────────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """
    Pre-activation residual block from architecture_torch.py.
    Used inside res_normN_AE.

    bn=True:  BN → ReLU → Conv → BN → ReLU → Conv + shortcut
    bn=False: ReLU → Conv → ReLU → Conv + shortcut

    No ReLU after addition — exact match to original.
    Shortcut projection (Conv1×1 + BN) added when shape changes.
    """
    expansion = 1

    def __init__(self, in_planes, planes, bn=False, stride=1):
        super(ResBlock, self).__init__()
        self.bn = bn

        if bn:
            self.bn0 = nn.BatchNorm2d(in_planes)

        self.conv1 = nn.Conv2d(in_planes, planes,
                               kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn0(x))
        else:
            out = F.relu(x)

        if self.bn:
            out = F.relu(self.bn1(self.conv1(out)))
        else:
            out = F.relu(self.conv1(out))

        out = self.conv2(out)
        out += self.shortcut(x)
        # No ReLU after addition — matches original exactly
        return out


# ── custom_AE ─────────────────────────────────────────────────────────────────
class custom_AE(nn.Module):
    """
    Custom AutoEncoder decoder — primary attacker model used in ResSFL.
    Exact port of custom_AE from architecture_torch.py.

    Architecture is computed dynamically from input/output dimensions:
    upsampling_num = int(log2(output_dim / input_dim))

    For upsampling_num=2 (our case for both CIFAR-10 and MNIST):
      nc starts at input_nc (32)

      Loop (upsampling_num-1 = 1 iteration):
        Conv(nc → nc//2, 3×3, pad=1) → ReLU
        ConvTranspose(nc//2 → nc//2, 3×3, stride=2, pad=1, out_pad=1) → ReLU
        nc = nc//2   [32 → 16]

      Final block:
        Conv(nc → nc, 3×3, pad=1) → ReLU
        ConvTranspose(nc → output_nc, 3×3, stride=2, pad=1, out_pad=1)
        → Sigmoid (for output in [0,1])

    CIFAR-10 [B,32,8,8] → [B,3,32,32]:
      8→16→32 spatial, 32→16→3 channels

    MNIST [B,32,7,7] → [B,1,28,28]:
      7→14→28 spatial, 32→16→1 channels

    No BatchNorm — distinguishing feature vs custom_AE_bn.
    Activation = 'sigmoid' → output in [0,1] matching denormalized images.
    """

    def __init__(self, input_nc=32, output_nc=3,
                 input_dim=8, output_dim=32,
                 activation='sigmoid'):
        super(custom_AE, self).__init__()

        # Number of 2× upsampling steps needed
        upsampling_num = int(np.log2(output_dim / input_dim))

        model = []
        nc    = input_nc

        # ── Loop: all but the last upsampling step ────────────────────────
        for _ in range(upsampling_num - 1):
            # Spatial: keeps size | Channel: nc → nc//2
            model += [nn.Conv2d(nc, nc // 2, kernel_size=3,
                                stride=1, padding=1)]
            model += [nn.ReLU()]
            # Spatial: ×2 | Channel: stays nc//2
            model += [nn.ConvTranspose2d(nc // 2, nc // 2, kernel_size=3,
                                         stride=2, padding=1,
                                         output_padding=1)]
            model += [nn.ReLU()]
            nc = nc // 2

        # ── Final block: last upsampling step ─────────────────────────────
        if upsampling_num >= 1:
            # nc after loop = input_nc / 2^(upsampling_num-1)
            final_nc = input_nc // (2 ** (upsampling_num - 1))

            model += [nn.Conv2d(final_nc, final_nc,
                                kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            # Final 2× upsample to output resolution with output_nc channels
            model += [nn.ConvTranspose2d(final_nc, output_nc,
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1)]
            if activation == 'sigmoid':
                model += [nn.Sigmoid()]
            elif activation == 'tanh':
                model += [nn.Tanh()]

        else:
            # upsampling_num == 0: input and output same spatial size
            # Just apply two convolutions to change channel count
            model += [nn.Conv2d(input_nc, input_nc,
                                kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]
            model += [nn.Conv2d(input_nc, output_nc,
                                kernel_size=3, stride=1, padding=1)]
            if activation == 'sigmoid':
                model += [nn.Sigmoid()]
            elif activation == 'tanh':
                model += [nn.Tanh()]

        self.m = nn.Sequential(*model)

    def forward(self, x):
        return self.m(x)


# ── AE factory function ───────────────────────────────────────────────────────
def build_ae(dataset='CIFAR10', activation='sigmoid'):
    """
    Builds the custom_AE decoder matched to the shallow ClientModel's
    smashed data shape for the given dataset.

    CIFAR-10: smashed [B,32,8,8]  → input_nc=32, input_dim=8,
              target  [B,3,32,32] → output_nc=3,  output_dim=32
    MNIST:    smashed [B,32,7,7]  → input_nc=32, input_dim=7,
              target  [B,1,28,28] → output_nc=1,  output_dim=28

    The AE is initialized with Xavier weights following the original.
    """
    if dataset == 'CIFAR10':
        ae = custom_AE(input_nc=32, output_nc=3,
                       input_dim=8, output_dim=32,
                       activation=activation)
    elif dataset == 'MNIST':
        ae = custom_AE(input_nc=32, output_nc=1,
                       input_dim=7, output_dim=28,
                       activation=activation)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Xavier initialization — from MIA_torch.init_weights()
    ae.apply(xavier_init)
    return ae