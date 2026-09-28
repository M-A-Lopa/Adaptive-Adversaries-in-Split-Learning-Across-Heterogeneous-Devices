"""
Paper-exact re-implementation of:

    N. D. Pham, K. T. Phan, N. Chilamkurti,
    "Enhancing Accuracy-Privacy Trade-off in Differentially Private Split Learning,"
    arXiv:2310.14434 (v3, Oct 2024) / IEEE TETCI vol. 9, no. 1.

Components (section of the paper in brackets):

  1. sigma_from_epsilon / GaussianDPNoise        [II-C, Eq. 3]
       Clamp the layer output to [0, 1] (1-sensitive), then add N(0, sigma^2)
       element-wise with sigma^2 = 2 s^2 log(1.25/delta) / eps^2.

  2. DPClientModel                               [IV-B, IV-C, V-B]
       Wraps a PyramidCNN / KAGN client and injects the Gaussian mechanism at
       ANY local layer ("Input", "Conv(1)", "ReLU(1)", "MaxP(1)", ...).
       Default = the split layer, which the paper recommends.

  3. ResizedDPClientModel + build_full_server    [IV-D, Eq. 7, V-C]
       h~ = (g o f) o (f' o F): the client appends one ConvTranspose2d (f')
       that maps the noisy smashed data back to the input size; the server
       runs the FULL model (g o f) on it.

  4. NoiseReviewServer                           [III-B, Alg. 2, Eq. 5-6]
       Server duplicates the incoming smashed data, adds extra noise with
       sigma_hat^2 = sigma_j^2 - sigma_i^2 (sigma_j = noisiest client), trains
       on the concatenation, and slices the split-layer gradient back to the
       original batch size before returning it.

The multi-client sequential trainer (Alg. 1) lives in
all_split_learning/pham_multiclient_sl.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════
# 1. Gaussian mechanism (Eq. 3)
# ════════════════════════════════════════════════════════════════════

def sigma_from_epsilon(epsilon, delta=1e-5, sensitivity=1.0):
    """sigma = sqrt(2 s^2 log(1.25/delta)) / eps.  epsilon=None -> no DP (sigma=0)."""
    if epsilon is None:
        return 0.0
    assert epsilon > 0, "epsilon must be > 0"
    assert 0 < delta < 1, "delta must be in (0, 1)"
    return math.sqrt(2.0 * (sensitivity ** 2) * math.log(1.25 / delta)) / epsilon


class GaussianDPNoise(nn.Module):

    def __init__(self, epsilon=None, delta=1e-5, sensitivity=1.0,
                 clamp=True, noise_at_eval=True):
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.clamp = clamp
        self.noise_at_eval = noise_at_eval
        self.active = True          # False -> clamp only (deterministic part of the mechanism)
        self.sigma = sigma_from_epsilon(epsilon, delta, sensitivity)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.sigma = sigma_from_epsilon(epsilon, self.delta, self.sensitivity)

    @property
    def enabled(self):
        return self.epsilon is not None

    def forward(self, x):
        if not self.enabled:
            return x
        if self.clamp:
            x = torch.clamp(x, 0.0, 1.0)
        if self.active and (self.training or self.noise_at_eval):
            x = x + torch.randn_like(x) * self.sigma
        return x

    def _hook(self, module, inputs, output):
        return self(output)

    def extra_repr(self):
        return f"epsilon={self.epsilon}, delta={self.delta}, sigma={self.sigma:.4f}"


# ════════════════════════════════════════════════════════════════════
# 2. Client with noise at any local layer (Sec. IV)
# ════════════════════════════════════════════════════════════════════

_DATASET_STATS = {
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'MNIST':   ((0.1307,), (0.3081,)),
}


def _layer_tag(module):
    """Short name used in the paper's plots: Conv, ReLU, MaxP, ..."""
    name = type(module).__name__
    if name.startswith('KAGNConv'):
        return 'KAGN'
    return {'Conv2d': 'Conv', 'BatchNorm2d': 'BN', 'ReLU': 'ReLU',
            'MaxPool2d': 'MaxP', 'AdaptiveAvgPool2d': 'AvgP',
            'SiLU': 'SiLU'}.get(name, name)


def flatten_client_layers(client_model):
    layers = []
    for g, block in enumerate(client_model.client_layers, start=1):
        children = list(block.children()) if isinstance(block, nn.Sequential) else [block]
        for m in children:
            layers.append((f"{_layer_tag(m)}({g})", m))
    return layers


class DPClientModel(nn.Module):

    def __init__(self, base_client, epsilon=None, delta=1e-5,
                 noise_location='split', dataset='CIFAR10', noise_at_eval=True):
        super().__init__()
        self.client_layers = base_client.client_layers          # registered FIRST
        self.noise = GaussianDPNoise(epsilon, delta, noise_at_eval=noise_at_eval)
        self.dataset = dataset
        named = flatten_client_layers(base_client)
        self.layer_names = [n for n, _ in named]

        if noise_location == 'split':
            noise_location = self.layer_names[-1]
        if noise_location not in self.available_locations():
            raise ValueError(f"noise_location '{noise_location}' not in {self.available_locations()}")
        self.noise_location = noise_location

        if noise_location != 'Input':
            target = named[self.layer_names.index(noise_location)][1]
            target.register_forward_hook(self.noise._hook)    # bound method -> deepcopy-safe

        mean, std = _DATASET_STATS[dataset]
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1), persistent=False)

    def available_locations(self):
        return ['Input'] + self.layer_names

    @property
    def sigma(self):
        return self.noise.sigma

    @property
    def epsilon(self):
        return self.noise.epsilon

    def set_epsilon(self, epsilon):
        self.noise.set_epsilon(epsilon)

    def _input_noise(self, x):
        # images are normalised by the loader; the paper's [0,1] clamp is meant
        # for pixel space, so go to [0,1], apply the mechanism, and come back
        x01 = self.noise(x * self.std + self.mean)
        return (x01 - self.mean) / self.std

    def forward(self, x):
        if self.noise_location == 'Input' and self.noise.enabled:
            x = self._input_noise(x)
        return self.client_layers(x)

    def clean_forward(self, x):
        prev = self.noise.active
        self.noise.active = False
        try:
            return self.forward(x)
        finally:
            self.noise.active = prev


# ════════════════════════════════════════════════════════════════════
# 3. Smashed-data resizing with ConvTranspose (Sec. IV-D, Eq. 7)
# ════════════════════════════════════════════════════════════════════

def _deconv_geometry(in_size, out_size):

    if in_size > out_size:
        raise ValueError(f"smashed spatial size {in_size} > input size {out_size}; cannot up-sample")
    stride = out_size // in_size
    kernel = out_size - (in_size - 1) * stride
    return kernel, stride


class ResizedDPClientModel(DPClientModel):
   

    def __init__(self, base_client, input_shape, epsilon=None, delta=1e-5,
                 dataset='CIFAR10', noise_at_eval=True):
        super().__init__(base_client, epsilon, delta, 'split', dataset, noise_at_eval)
        c_in, h_in, w_in = input_shape
        with torch.no_grad():
            was = self.training
            self.eval()
            s = self.client_layers(torch.zeros(1, *input_shape))
            self.train(was)
        _, c_s, h_s, w_s = s.shape
        kh, sh = _deconv_geometry(h_s, h_in)
        kw, sw = _deconv_geometry(w_s, w_in)
        self.f_prime = nn.ConvTranspose2d(c_s, c_in, kernel_size=(kh, kw), stride=(sh, sw))
        self.smashed_shape_before = (c_s, h_s, w_s)
        self.output_shape = input_shape

    def forward(self, x):
        return self.f_prime(super().forward(x))


class NoiseFreeView(nn.Module):
    """
    Shares the weights of a trained DP client but runs it with the noise off
    (clamp kept). This is the model a WHITE-BOX attacker optimises through:
    it knows the weights and the mechanism, but not the random noise draw.
    """

    def __init__(self, dp_client):
        super().__init__()
        self.dp = dp_client

    def forward(self, x):
        return self.dp.clean_forward(x)


class FullModelServer(nn.Module):
    """g o f : the complete (unsplit) network, run on the up-sampled smashed data."""

    def __init__(self, full_client_part, server_part):
        super().__init__()
        self.f = full_client_part
        self.g = server_part

    def forward(self, z):
        return self.g(self.f(z))


def build_full_server(model_name, cut_layer, in_channels, num_classes, degree=3):
    """
    Server side of Eq. 7. The server gets a fresh copy of the client layers (f)
    plus its usual layers (g), i.e. the whole PyramidCNN / KAGN.
    """
    if model_name == 'PyramidCNN':
        from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
        f = PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels)
        g = PyramidCNNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels)
    elif model_name == 'KAGN':
        from all_model.kagn_models import KAGNClientModel, KAGNServerModel
        f = KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=degree)
        g = KAGNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels, degree=degree)
    else:
        raise ValueError("model_name must be 'PyramidCNN' or 'KAGN'")
    return FullModelServer(f, g)


# ════════════════════════════════════════════════════════════════════
# 4. Server-side review of noise distributions (Sec. III-B, Alg. 2)
# ════════════════════════════════════════════════════════════════════

class NoiseReviewServer:

    def __init__(self, client_sigmas, enabled=True):
        self.enabled = enabled
        self.sigma_max = max(client_sigmas) if client_sigmas else 0.0

    def sigma_hat(self, sigma_i):
        return math.sqrt(max(self.sigma_max ** 2 - sigma_i ** 2, 0.0))

    def prepare(self, smashed, labels, sigma_i):
        """Returns (server_input, server_labels, original_batch_size)."""
        B = smashed.size(0)
        s_hat = self.sigma_hat(sigma_i)
        if not self.enabled or s_hat == 0.0:
            return smashed, labels, B
        dup = smashed.detach() + torch.randn_like(smashed) * s_hat
        return torch.cat([smashed, dup], dim=0), torch.cat([labels, labels], dim=0), B

    @staticmethod
    def slice_gradient(grad, B):
        return grad[:B]


# ════════════════════════════════════════════════════════════════════
# Convenience builders
# ════════════════════════════════════════════════════════════════════

def build_base_split(model_name, cut_layer, in_channels, num_classes, degree=3):
    if model_name == 'PyramidCNN':
        from all_model.pyramid_cnn import PyramidCNNClientModel, PyramidCNNServerModel
        return (PyramidCNNClientModel(cut_layer=cut_layer, in_channels=in_channels),
                PyramidCNNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels))
    if model_name == 'KAGN':
        from all_model.kagn_models import KAGNClientModel, KAGNServerModel
        return (KAGNClientModel(cut_layer=cut_layer, in_channels=in_channels, degree=degree),
                KAGNServerModel(cut_layer=cut_layer, num_classes=num_classes, in_channels=in_channels, degree=degree))
    raise ValueError("model_name must be 'PyramidCNN' or 'KAGN'")


def upgrade_client_state(state):

    out = {}
    for k, v in state.items():
        if k.startswith('inner.'):
            k = k[len('inner.'):]
        if k.startswith('layers.') or k in ('mean', 'std'):
            continue
        if k.startswith('base.'):
            k = k[len('base.'):]
        out[k] = v
    return out


def dpsl_run_name(model_cut_layer, epsilon, noise_location='split', resize=False):
    """Shared naming for checkpoints / result folders across the DP-SL runners."""
    import re
    loc = 'resize' if resize else re.sub(r'[()]', '', noise_location)
    return f"dpsl_{loc}_cut{model_cut_layer}_" + ('nonoise' if epsilon is None else f"eps{epsilon:g}")


def build_dp_split(model_name, cut_layer, dataset, num_classes, epsilon=None, delta=1e-5,
                   noise_location='split', resize=False, degree=3):
    
    in_ch = 1 if dataset == 'MNIST' else 3
    hw = 28 if dataset == 'MNIST' else 32
    base_c, base_s = build_base_split(model_name, cut_layer, in_ch, num_classes, degree)
    if not resize:
        return DPClientModel(base_c, epsilon, delta, noise_location, dataset), base_s
    client = ResizedDPClientModel(base_c, (in_ch, hw, hw), epsilon, delta, dataset)
    server = build_full_server(model_name, cut_layer, in_ch, num_classes, degree)
    return client, server