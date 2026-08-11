import torch.nn as nn


class ClientModel(nn.Module):

    def __init__(self, in_channels=3):
        super(ClientModel, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x  


class ServerModel(nn.Module):

    def __init__(self, num_classes=10):
        super(ServerModel, self).__init__()

        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))

        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, num_classes))

    def forward(self, smashed_data):
        x = self.conv3(smashed_data)
        x = self.fc(x)
        return x