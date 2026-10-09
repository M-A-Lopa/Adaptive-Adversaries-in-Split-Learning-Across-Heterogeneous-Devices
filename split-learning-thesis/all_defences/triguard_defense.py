import copy

import torch
import torch.nn as nn

from all_defences.splitfss_defense import FixedPoint, CryptoProvider
from all_defences.r3elu_defense import sample_laplace

COMPONENT_SOURCES = {
    'secret_sharing': 'SplitFSS (Make Split, not Hijack): additive secret sharing over the full ring, two non-colluding servers',
    'backward_laplace': 'R3eLU (Secure Split Learning): Laplace noise on the cut-layer gradients returned to the client',
    'private_heads': 'SplitML: private, non-averaged classifier layers per client; only the lower layers are averaged',
}


class ShareView:
    def __init__(self, mode='full_ring', precision_fractional=5, bounded_scale=1.0):
        assert mode in ('full_ring', 'bounded', 'plain')
        self.mode = mode
        self.fp = FixedPoint(10, precision_fractional)
        self.dealer = CryptoProvider()
        self.bounded_scale = bounded_scale

    def server_view(self, z):
        if self.mode == 'plain':
            return z
        if self.mode == 'bounded':
            return z + (torch.rand_like(z) * 2 - 1) * self.bounded_scale * z.detach().std()
        flat = z.detach().reshape(z.shape[0], -1)
        share0, _ = self.dealer.share(self.fp.encode(flat))
        return self.fp.decode(share0).float().reshape(z.shape).to(z.device)


class BackwardLaplace:
    def __init__(self, num_features, epsilon=1.0, C=10.0, K=None, sensitivity='global', enabled=True):
        assert sensitivity in ('global', 'per_coord')
        self.num_features = int(num_features)
        self.epsilon = float(epsilon)
        self.C = float(C)
        self.K = int(K) if K is not None else max(1, self.num_features // 2)
        self.sensitivity = sensitivity
        self.enabled = enabled

    def scale(self):
        sens = 2.0 * self.K * self.C if self.sensitivity == 'global' else 2.0 * self.C
        return sens / (self.epsilon / 2.0)

    @torch.no_grad()
    def __call__(self, grad):
        if not self.enabled:
            return grad
        flat = grad.reshape(grad.shape[0], -1).clamp(-self.C, self.C)
        return (flat + sample_laplace(flat.shape, self.scale(), grad.device)).reshape(grad.shape)


class GradientNoiseServer(nn.Module):
    def __init__(self, server, noise):
        super().__init__()
        self.server = server
        self.noise = noise

    def forward(self, z):
        if torch.is_grad_enabled() and z.requires_grad and self.noise.enabled:
            z.register_hook(self.noise)
        return self.server(z)


def _last_linear_path(module):
    path = None
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            path = name
    if path is None:
        raise ValueError("server model has no nn.Linear classifier layer")
    return path


def _set_submodule(root, path, new):
    parts = path.split('.')
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new
    else:
        setattr(parent, parts[-1], new)


def split_server(server):
    path = _last_linear_path(server)
    head = copy.deepcopy(server.get_submodule(path))
    backbone = copy.deepcopy(server)
    _set_submodule(backbone, path, nn.Identity())
    return backbone, head


def fedavg(modules):
    states = [m.state_dict() for m in modules]
    avg = {}
    for k, v in states[0].items():
        if v.dtype.is_floating_point:
            avg[k] = torch.stack([s[k] for s in states]).mean(0)
        else:
            avg[k] = v.clone()
    for m in modules:
        m.load_state_dict(avg)


class TriGuardSL:
    name = 'TriGuard-SL'

    def __init__(self, num_features, num_clients=3, share_mode='full_ring', backward_noise=True,
                 private_heads=True, epsilon=1.0, C=10.0, sensitivity='global'):
        self.num_clients = int(num_clients)
        self.view = ShareView(share_mode)
        self.noise = BackwardLaplace(num_features, epsilon=epsilon, C=C, sensitivity=sensitivity,
                                     enabled=backward_noise)
        self.private_heads = private_heads
        self.config = dict(num_clients=num_clients, share_mode=share_mode, backward_noise=backward_noise,
                           private_heads=private_heads, epsilon=epsilon, C=C, sensitivity=sensitivity)

    def server_view(self, z):
        return self.view.server_view(z)

    def wrap_server(self, server):
        return GradientNoiseServer(server, self.noise)


class TriGuardTrainer:
    def __init__(self, defense, client, server, shard_loaders, device, lr):
        self.defense = defense
        self.device = device
        self.K = len(shard_loaders)
        self.loaders = shard_loaders
        self.backbone, head = split_server(server)
        self.bottoms = [copy.deepcopy(client) for _ in range(self.K)]
        self.heads = [copy.deepcopy(head) for _ in range(self.K)]
        self.opt_backbone = torch.optim.Adam(self.backbone.parameters(), lr=lr)
        self.opt_bottoms = [torch.optim.Adam(b.parameters(), lr=lr) for b in self.bottoms]
        self.opt_heads = [torch.optim.Adam(h.parameters(), lr=lr) for h in self.heads]
        self.criterion = nn.CrossEntropyLoss()

    def _step(self, k, x, y):
        bottom, head = self.bottoms[k], self.heads[k]
        self.opt_bottoms[k].zero_grad()
        self.opt_heads[k].zero_grad()
        self.opt_backbone.zero_grad()
        z = bottom(x)
        z_leaf = z.detach().requires_grad_(True)
        loss = self.criterion(head(self.backbone(z_leaf)), y)
        loss.backward()
        self.opt_backbone.step()
        self.opt_heads[k].step()
        z.backward(self.defense.noise(z_leaf.grad))
        self.opt_bottoms[k].step()
        return loss.item()

    def train_round(self, poison_fn=None, malicious=None):
        for m in self.bottoms + self.heads + [self.backbone]:
            m.train()
        for k in range(self.K):
            for b in self.loaders[k]:
                x, y = b[0].to(self.device), b[1].to(self.device)
                if poison_fn is not None and k == malicious:
                    x, y = poison_fn(x, y)[:2]
                self._step(k, x, y)
        fedavg(self.bottoms)
        if not self.defense.private_heads:
            fedavg(self.heads)

    @torch.no_grad()
    def evaluate(self, k, loader, trigger_fn=None, target=None):
        bottom, head = self.bottoms[k], self.heads[k]
        for m in (bottom, head, self.backbone):
            m.eval()
        correct = total = hit = trig_total = 0
        for b in loader:
            x, y = b[0].to(self.device), b[1].to(self.device)
            out = head(self.backbone(bottom(x)))
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
            if trigger_fn is not None:
                rows = y != target
                if rows.any():
                    xt = trigger_fn(x[rows])
                    hit += head(self.backbone(bottom(xt))).argmax(1).eq(target).sum().item()
                    trig_total += int(rows.sum())
        for m in (bottom, head, self.backbone):
            m.train()
        acc = 100.0 * correct / max(total, 1)
        asr = 100.0 * hit / max(trig_total, 1) if trigger_fn is not None else None
        return acc, asr