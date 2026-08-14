import torch
import torch.nn as nn


def get_pyramid_cnn_blocks(in_channels=3):
    return [
        nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ),
        nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
    ]


class PyramidCNNClientModel(nn.Module):
    def __init__(self, cut_layer=4, in_channels=3):
        super(PyramidCNNClientModel, self).__init__()
        all_blocks = get_pyramid_cnn_blocks(in_channels=in_channels)
        
        if not (1 <= cut_layer <= len(all_blocks)):
            raise ValueError(f"cut_layer must be between 1 and {len(all_blocks)}, got {cut_layer}")

        self.client_layers = nn.Sequential(*all_blocks[:cut_layer])

    def forward(self, x):
        return self.client_layers(x)


class PyramidCNNServerModel(nn.Module):
    def __init__(self, cut_layer=4, num_classes=10, in_channels=3):
        super(PyramidCNNServerModel, self).__init__()

        all_blocks = get_pyramid_cnn_blocks(in_channels=in_channels)

        if not (1 <= cut_layer <= len(all_blocks)):
            raise ValueError(
                f"cut_layer must be between 1 and {len(all_blocks)}, got {cut_layer}"
            )

        self.server_layers = nn.Sequential(*all_blocks[cut_layer:])


        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, smashed_data):
        x = self.server_layers(smashed_data)
        x = self.fc(x)
        return x


def build_pyramid_cnn_split_models(cut_layer=4, in_channels=3, num_classes=10):
    
    client = PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels)
    server = PyramidCNNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels)
    return client, server