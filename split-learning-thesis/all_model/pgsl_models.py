import torch
import torch.nn as nn
from config import Config
from all_defences.pgsl_defense import PGSLProximalRecoveryBlock, ConvolutionSumFusion
from all_model.kagn_models import KAGNClientModel
from all_model.pyramid_cnn import PyramidCNNClientModel
from all_model.models import ClientModel


def build_pgsl_client(original_in_channels=3):
    pgsl_in_channels = 4 * original_in_channels + 1

    if Config.MODEL_NAME == "KAGN":
        return KAGNClientModel(cut_layer=Config.CUT_LAYER, in_channels=pgsl_in_channels, degree=Config.DEGREE)
    elif Config.MODEL_NAME == "PyramidCNN":
        return PyramidCNNClientModel(cut_layer=Config.CUT_LAYER, in_channels=pgsl_in_channels)
    else:
        return ClientModel(in_channels=pgsl_in_channels)


class PGSLClientModel(nn.Module):
    def __init__(self, original_in_channels=1):
        super(PGSLClientModel, self).__init__()
        self.base = build_pgsl_client(original_in_channels)

    def forward(self, x):
        return self.base(x)


class PGSLServerModel(nn.Module):
    def __init__(self, num_classes=10, smashed_channels=None, smashed_shape=None):
        super(PGSLServerModel, self).__init__()
        if smashed_channels is None:
            if smashed_shape is None:
                raise ValueError(
                    "PGSLServerModel needs smashed_channels or smashed_shape -- "
                    "KAGN/PyramidCNN change the client's output channel count "
                    "with Config.CUT_LAYER, so it can no longer be hardcoded to 32."
                )
            smashed_channels = smashed_shape[1]

        self.recovery = PGSLProximalRecoveryBlock(mu=0.55)
        self.fusion   = ConvolutionSumFusion(smashed_channels)

        self.stream_a = self._make_backbone(smashed_channels)
        self.stream_r = self._make_backbone(smashed_channels)
        self.stream_f = self._make_backbone(smashed_channels)

        self.classifier_a = self._make_classifier(num_classes)
        self.classifier_r = self._make_classifier(num_classes)
        self.classifier_f = self._make_classifier(num_classes)

    def _make_backbone(self, in_channels):
        return nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                             nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))

    def _make_classifier(self, num_classes):
        return nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
                             nn.Dropout(p=0.3), nn.Linear(256, num_classes))

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