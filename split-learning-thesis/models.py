# Defines client-side and server-side sub-models for Vanilla Split Learning
#
# Architecture (CIFAR-10):
# Client: Conv1 → Conv2 → [smashed data sent to server]
# Server: Conv3 → AdaptivePool → FC1 → FC2 → Output


import torch.nn as nn


class ClientModel(nn.Module):
    
    def __init__(self, in_channels=3):
        super(ClientModel, self).__init__()

        # Conv1: first feature extraction block
        # CIFAR-10 input: 3 x 32 x 32
        # After Conv1 + MaxPool: 16 x 16 x 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Conv2: second feature extraction block (CUT LAYER)
        # After Conv2 + MaxPool: 32 x 8 x 8
        # This output is the smashed data transmitted to the server
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  # Smashed data — shape: [batch_size, 32, 8, 8] for CIFAR-10


class ServerModel(nn.Module):
    """
    AdaptiveAvgPool2d maps ANY smashed data spatial size to fixed 4x4.
    This means the server handles different client cut depths automatically.
    Critical for heterogeneous setup in Week 9-11.
    """

    def __init__(self, num_classes=10):
        super(ServerModel, self).__init__()

        # Conv3: deeper feature extraction on server side
        # Input smashed data: 32 x 8 x 8
        # After Conv3 + AdaptivePool: 64 x 4 x 4 (always, regardless of input size)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Fixed 4x4 output regardless of smashed data size
        )

        # Fully connected classification head
        # 64 * 4 * 4 = 1024 input features (fixed due to AdaptiveAvgPool2d)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),           # Regularization — reduces overfitting
            nn.Linear(256, num_classes)  # Final output: 10 classes for CIFAR-10/MNIST
        )

    def forward(self, smashed_data):
        x = self.conv3(smashed_data)
        x = self.fc(x)
        return x