import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim_compute(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2,
                         groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=window_size // 2,
                         groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12   + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class WindowedSSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(WindowedSSIM, self).__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.channel      = 1
        self.window       = _create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and \
                self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window  = window
            self.channel = channel

        return _ssim_compute(img1, img2, window, self.window_size, channel, self.size_average)


def denormalize(x, dataset):
    if dataset == 'CIFAR10':

        mean = torch.tensor([0.4914, 0.4822, 0.4465],
                            device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.2023, 0.1994, 0.2010],
                            device=x.device).view(1, 3, 1, 1)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    elif dataset == 'MNIST':

        mean = torch.tensor([0.1307], device=x.device).view(1, 1, 1, 1)
        std  = torch.tensor([0.3081], device=x.device).view(1, 1, 1, 1)
        return torch.clamp(x * std + mean, 0.0, 1.0)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use 'CIFAR10' or 'MNIST'.")


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            m.bias.data.zero_()