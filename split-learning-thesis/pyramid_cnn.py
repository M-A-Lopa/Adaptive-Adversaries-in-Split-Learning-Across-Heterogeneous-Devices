# pyramid_cnn.py
# PyramidCNN Model for Split Learning on Heterogeneous Devices

import torch
import torch.nn as nn


# ─────────────────────────────────────────
# A Single Convolution Block
# Conv → BatchNorm → ReLU
# ─────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────
# PyramidCNN — Main Model
# ─────────────────────────────────────────
class PyramidCNN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(PyramidCNN, self).__init__()

        # ── Client-side layers (runs on the device) ──
        # Layer 1: 3 → 32 channels (this much is done even by the weakest device)
        self.layer1 = ConvBlock(in_channels, 32)

        # Layer 2: 32 → 64 channels
        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)  # image size is halved
        )

        # Layer 3: 64 → 128 channels
        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ── Server-side layers (runs on the server) ──
        # Layer 4: 128 → 256 channels
        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Layer 5: 256 → 512 channels (widest part of the pyramid)
        self.layer5 = nn.Sequential(
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1))  # handles any input size
        )

        # ── Classifier ──
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────
# Separate Client and Server Models for Split Learning
# ─────────────────────────────────────────

class PyramidCNN_Client(nn.Module):
    """
    Runs on the device — how much runs is determined by cut_layer.
    cut_layer=1 → layer1 only
    cut_layer=2 → layer1 + layer2
    cut_layer=3 → layer1 + layer2 + layer3
    """
    def __init__(self, cut_layer=2, in_channels=3):
        super(PyramidCNN_Client, self).__init__()

        self.cut_layer = cut_layer

        self.layer1 = ConvBlock(in_channels, 32)

        self.layer2 = nn.Sequential(
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2)
        )

        self.layer3 = nn.Sequential(
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.layer1(x)
        if self.cut_layer >= 2:
            x = self.layer2(x)
        if self.cut_layer >= 3:
            x = self.layer3(x)
        return x  # this smashed data will be sent to the server


class PyramidCNN_Server(nn.Module):
    """
    Runs on the server — receives smashed data from the client and does the rest.
    The server knows which channel to start from based on cut_layer.
    """
    def __init__(self, cut_layer=2, num_classes=10):
        super(PyramidCNN_Server, self).__init__()

        # input channel count differs depending on cut_layer
        in_ch_map = {1: 32, 2: 64, 3: 128}
        in_ch = in_ch_map[cut_layer]

        layers = []

        if cut_layer <= 1:
            layers.append(nn.Sequential(ConvBlock(32, 64), nn.MaxPool2d(2, 2)))
        if cut_layer <= 2:
            layers.append(nn.Sequential(ConvBlock(in_ch if cut_layer == 2 else 64, 128), nn.MaxPool2d(2, 2)))
            in_ch = 128

        layers.append(nn.Sequential(ConvBlock(in_ch if cut_layer == 3 else 128, 256), nn.MaxPool2d(2, 2)))
        layers.append(nn.Sequential(ConvBlock(256, 512), nn.AdaptiveAvgPool2d((1, 1))))

        self.server_layers = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.server_layers(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────
# Test — check that everything works correctly
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("PyramidCNN Test")
    print("=" * 50)

    # Fake input: batch=4, channel=3, image=32x32 (like CIFAR-10)
    dummy_input = torch.randn(4, 3, 32, 32)

    # ── Test 1: Full Model ──
    full_model = PyramidCNN(num_classes=10)
    output = full_model(dummy_input)
    print(f"✅ Full Model Output Shape   : {output.shape}")  # should be (4, 10)

    # ── Test 2: Split Model (cut_layer=1) — weak device ──
    client1 = PyramidCNN_Client(cut_layer=1)
    smashed1 = client1(dummy_input)
    print(f"✅ Cut Layer 1 Smashed Shape : {smashed1.shape}")  # (4, 32, 32, 32)

    # ── Test 3: Split Model (cut_layer=2) — medium device ──
    client2 = PyramidCNN_Client(cut_layer=2)
    smashed2 = client2(dummy_input)
    print(f"✅ Cut Layer 2 Smashed Shape : {smashed2.shape}")  # (4, 64, 16, 16)

    # ── Test 4: Split Model (cut_layer=3) — strong device ──
    client3 = PyramidCNN_Client(cut_layer=3)
    smashed3 = client3(dummy_input)
    print(f"✅ Cut Layer 3 Smashed Shape : {smashed3.shape}")  # (4, 128, 8, 8)

    print("=" * 50)
    print("All good! PyramidCNN creation complete.")
    print("=" * 50)