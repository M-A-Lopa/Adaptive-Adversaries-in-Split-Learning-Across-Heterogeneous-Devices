import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradientDefenseFn(torch.autograd.Function):
   
    @staticmethod
    def forward(ctx, z, flip_prob, noise_scale, clip_norm):
        ctx.flip_prob = flip_prob
        ctx.noise_scale = noise_scale
        ctx.clip_norm = clip_norm
        return z 

    @staticmethod
    def backward(ctx, grad_output):
        grad_norm = torch.norm(
            grad_output.view(grad_output.size(0), -1), dim=1, keepdim=True
        ).view(-1, 1, 1, 1) + 1e-8
        direction = grad_output / grad_norm

        clipped_norm = torch.clamp(grad_norm, max=ctx.clip_norm)
        noisy_norm = torch.clamp(
            clipped_norm + torch.randn_like(clipped_norm) * ctx.noise_scale * ctx.clip_norm,
            min=1e-8
        )

        mask = (torch.rand_like(direction) > ctx.flip_prob).float() * 2.0 - 1.0
        perturbed_direction = direction * mask
        perturbed_direction = perturbed_direction + torch.randn_like(perturbed_direction) * ctx.noise_scale

        return perturbed_direction * noisy_norm, None, None, None


class DefendedClientModel(nn.Module):
    def __init__(self, base_client, in_channels, image_size, device,
             gpi_scale_range=(0.5, 2.0), vel_momentum=0.1, clip_thresh=3.0,
             use_augmentation=True, use_vel_norm=True, use_persample_norm=True,
             use_clip=True, use_gpi=True):
        super().__init__()
        self.base_client = base_client
        self.device = device
        self.vel_momentum = vel_momentum
        self.vel_eps = 1e-5
        self.clip_thresh = clip_thresh

        # Ablation switches -- each defaults True, so existing call sites
        # that don't pass these get the original full-defense behavior.
        self.use_augmentation = use_augmentation
        self.use_vel_norm = use_vel_norm
        self.use_persample_norm = use_persample_norm
        self.use_clip = use_clip
        self.use_gpi = use_gpi

        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, image_size, image_size, device=device)
            z = self.base_client(dummy)
        _, num_channels, h, w = z.shape

        self.register_buffer('running_var', torch.ones(num_channels))
        self._gpi_shape = (num_channels, h, w)
        self._gpi_scale_range = gpi_scale_range
        self._gpi_device = device

        self.register_buffer('gpi_mask', self._generate_gpi_mask())
        self.batches_since_resample = 0
        self.last_raw_z = None
        self.last_sent_z = None

    def _generate_gpi_mask(self):
        c, h, w = self._gpi_shape
        signs = torch.randint(0, 2, (c, h, w), device=self._gpi_device).float() * 2 - 1
        magnitudes = torch.empty(c, h, w, device=self._gpi_device).uniform_(*self._gpi_scale_range)
        return signs * magnitudes

    def resample_gpi_mask(self):
        with torch.no_grad():
            self.gpi_mask.copy_(self._generate_gpi_mask())
        self.batches_since_resample = 0

    def forward(self, x):
        if self.training and self.use_augmentation:
            b, c, h, w = x.shape
            x_padded = F.pad(x, (2, 2, 2, 2), mode='reflect')
            crop_h = torch.randint(0, 5, (1,)).item()
            crop_w = torch.randint(0, 5, (1,)).item()
            x = x_padded[:, :, crop_h:crop_h+h, crop_w:crop_w+w]

            x = x + torch.randn_like(x) * 0.05

        z = self.base_client(x)
        self.last_raw_z = z

        if self.use_vel_norm:
            batch_var = z.var(dim=(0, 2, 3), unbiased=False) + self.vel_eps
            if self.training:
                with torch.no_grad():
                    self.running_var.mul_(1 - self.vel_momentum).add_(self.vel_momentum * batch_var.detach())
                norm_factor = batch_var.sqrt()
            else:
                norm_factor = self.running_var.sqrt()
            z_normed = z / norm_factor.view(1, -1, 1, 1)
        else:
            z_normed = z

        if self.use_persample_norm:
            per_sample_var = z_normed.var(dim=1, keepdim=True, unbiased=False) + self.vel_eps
            z_normed = z_normed / per_sample_var.sqrt()

        if self.use_clip:
            z_norms = torch.norm(z_normed.view(z_normed.size(0), -1), p=2, dim=1, keepdim=True)
            z_norms = z_norms.view(-1, 1, 1, 1)
            max_norm = self.clip_thresh
            z_normed = z_normed * torch.clamp(max_norm / (z_norms + 1e-6), max=1.0)

        if self.use_gpi:
            z_sent = z_normed * self.gpi_mask.unsqueeze(0)
        else:
            z_sent = z_normed

        self.last_sent_z = z_sent

        if self.training:
            self.batches_since_resample += 1

        return z_sent

    def decorr_loss(self):
        if self.last_raw_z is None:
            raise RuntimeError("Call forward() before decorr_loss().")
        z = self.last_raw_z
        b, c, h, w = z.shape
        z_flat = z.view(b, c, -1).mean(dim=2)
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / max(b - 1, 1)
        identity = torch.eye(c, device=z.device)
        return torch.norm(cov - identity, p='fro') ** 2 / c

    def dcor_loss(self, x):
        if self.last_sent_z is None:
            raise RuntimeError("Call forward() before dcor_loss().")
        b = x.size(0)
        x_flat = x.view(b, -1)
        z_flat = self.last_sent_z.view(b, -1)

        dx = torch.cdist(x_flat, x_flat, p=2)
        dz = torch.cdist(z_flat, z_flat, p=2)

        A = dx - dx.mean(dim=0, keepdim=True) - dx.mean(dim=1, keepdim=True) + dx.mean()
        B = dz - dz.mean(dim=0, keepdim=True) - dz.mean(dim=1, keepdim=True) + dz.mean()

        dcov2_xz = (A * B).sum() / (b * b)
        dcov2_xx = (A * A).sum() / (b * b)
        dcov2_zz = (B * B).sum() / (b * b)

        dcor = torch.sqrt(dcov2_xz / (torch.sqrt(dcov2_xx * dcov2_zz) + 1e-8) + 1e-8)
        return dcor


class DefendedServerModel(nn.Module):
    def __init__(self, base_server, percentile=95, dropout_p=0.10, flip_prob=0.15,
             noise_scale=0.05, grad_clip_norm=1.0,
             use_grad_defense=True, use_server_sanitization=True):
        super().__init__()
        self.base_server = base_server
        self.percentile = percentile
        self.dropout_p = dropout_p
        self.flip_prob = flip_prob
        self.noise_scale = noise_scale
        self.grad_clip_norm = grad_clip_norm
        self.use_grad_defense = use_grad_defense
        self.use_server_sanitization = use_server_sanitization

    def forward(self, z):
        if self.training and self.use_grad_defense:
            z = _GradientDefenseFn.apply(z, self.flip_prob, self.noise_scale, self.grad_clip_norm)

        if self.use_server_sanitization:
            clip_val = torch.quantile(z.abs(), self.percentile / 100.0)
            z_clipped = torch.clamp(z, min=-clip_val, max=clip_val)
            z_out = F.dropout2d(z_clipped, p=self.dropout_p, training=self.training)
        else:
            z_out = z

        return self.base_server(z_out)