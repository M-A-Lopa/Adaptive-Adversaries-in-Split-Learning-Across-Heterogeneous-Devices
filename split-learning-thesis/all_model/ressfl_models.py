import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from all_defences.ressfl_defense import xavier_init


class ResBlock(nn.Module):

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
        
        return out

class custom_AE(nn.Module):
    def __init__(self, input_nc=32, output_nc=3,
                 input_dim=8, output_dim=32,
                 activation='sigmoid'):
        super(custom_AE, self).__init__()

        upsampling_num = int(np.log2(output_dim / input_dim))

        model = []
        nc    = input_nc

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

        if upsampling_num >= 1:

            final_nc = input_nc // (2 ** (upsampling_num - 1))

            model += [nn.Conv2d(final_nc, final_nc,
                                kernel_size=3, stride=1, padding=1)]
            model += [nn.ReLU()]

            model += [nn.ConvTranspose2d(final_nc, output_nc,
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1)]
            if activation == 'sigmoid':
                model += [nn.Sigmoid()]
            elif activation == 'tanh':
                model += [nn.Tanh()]

        else:

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

def build_ae(dataset='CIFAR10', activation='sigmoid'):

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

    ae.apply(xavier_init)
    return ae