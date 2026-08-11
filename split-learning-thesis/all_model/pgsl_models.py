import torch
import torch.nn as nn
from all_defences.pgsl_defense import (PGSLProximalRecoveryBlock, ConvolutionSumFusion)


class PGSLClientModel(nn.Module):
    def __init__(self, original_in_channels=1):
        super(PGSLClientModel, self).__init__()

        pgsl_in_channels = 4 * original_in_channels + 1

        self.conv1 = nn.Sequential(nn.Conv2d(pgsl_in_channels, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv2 = nn.Sequential( nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x 


class PGSLServerModel(nn.Module):
    def __init__(self, num_classes=10, smashed_channels=32):
        super(PGSLServerModel, self).__init__()

        self.recovery = PGSLProximalRecoveryBlock(mu=0.55)
        self.fusion   = ConvolutionSumFusion(smashed_channels)

        self.stream_a = self._make_backbone(smashed_channels)
        self.stream_r = self._make_backbone(smashed_channels)
        self.stream_f = self._make_backbone(smashed_channels)

        self.classifier_a = self._make_classifier(num_classes)
        self.classifier_r = self._make_classifier(num_classes)
        self.classifier_f = self._make_classifier(num_classes)

    def _make_backbone(self, in_channels):
        return nn.Sequential( nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))

    def _make_classifier(self, num_classes):
        
        return nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(256, num_classes))

    def forward(self, smashed_data, run_full_pipeline=True):

        features_a = self.stream_a(smashed_data)
        out_a      = self.classifier_a(features_a)

        if not run_full_pipeline:
            return out_a, None, None

        smashed_r  = self.recovery(smashed_data)
        features_r = self.stream_r(smashed_r)
        out_r      = self.classifier_r(features_r)

        smashed_f  = self.fusion(smashed_data, smashed_r)
        features_f = self.stream_f(smashed_f)
        out_f      = self.classifier_f(features_f)

        return out_a, out_r, out_f