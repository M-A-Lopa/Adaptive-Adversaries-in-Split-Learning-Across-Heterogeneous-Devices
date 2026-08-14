import torch
import torch.nn as nn
from .kagn_conv_v2 import KAGNConv2DLayerV2


def get_kagn_blocks(in_channels=3, degree=3):
    return [
        nn.Sequential(
            KAGNConv2DLayerV2(in_channels, 32, kernel_size=3, padding=1, stride=1, degree=degree),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ),
        nn.Sequential(
            KAGNConv2DLayerV2(32, 64, kernel_size=3, padding=1, stride=1, degree=degree),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ),
        nn.Sequential(
            KAGNConv2DLayerV2(64, 64, kernel_size=3, padding=1, stride=1, degree=degree)
        ),
        nn.Sequential(
            KAGNConv2DLayerV2(64, 32, kernel_size=3, padding=1, stride=1, degree=degree),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
    ]


class KAGNClientModel(nn.Module):
    def __init__(self, cut_layer=2, in_channels=3, degree=3):
        super(KAGNClientModel, self).__init__()
        all_blocks = get_kagn_blocks(in_channels=in_channels, degree=degree)
        
        if not (1 <= cut_layer <= len(all_blocks)):
            raise ValueError(f"cut_layer must be between 1 and {len(all_blocks)}, got {cut_layer}")

        self.client_layers = nn.Sequential(*all_blocks[:cut_layer])

    def forward(self, x):
        return self.client_layers(x)


class KAGNServerModel(nn.Module):
    def __init__(self, cut_layer=2, num_classes=10, in_channels=3, degree=3):
        super(KAGNServerModel, self).__init__()
        all_blocks = get_kagn_blocks(in_channels=in_channels, degree=degree)
        
        if not (1 <= cut_layer <= len(all_blocks)):
            raise ValueError(f"cut_layer must be between 1 and {len(all_blocks)}, got {cut_layer}")
        
        self.server_layers = nn.Sequential(*all_blocks[cut_layer:])
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, smashed_data):
        x = self.server_layers(smashed_data)
        x = self.fc(x)
        return x


def build_kagn_split_models(cut_layer=2, in_channels=3, degree=3, num_classes=10):
    client = KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=degree)
    server = KAGNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels, degree=degree)
    return client, server