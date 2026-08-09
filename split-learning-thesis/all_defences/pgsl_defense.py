# pgsl_defense.py
# PGSL Defense Modules
# Based on: "Get your Foes Fooled: Proximal Gradient Split Learning
# for Defense against Model Inversion Attacks on IoMT data"
#
# Key corrections from original:
# 1. mu = 0.55 (paper Fig 3 sensitivity analysis — optimal value)
# 2. Gradient formula: grad_f = bar_m - smashed (no factor of 2)
# 3. iterations = 10 minimum for SVT convergence
# 4. Fixed requires_grad_() in-place usage
# 5. Fixed indentation errors

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGSLDefenseModules:
    """
    Preprocessing modules applied at client side before transmitting smashed data.
    Section IV-A and IV-B of the paper.
    """

    @staticmethod
    def generate_faithful_saliency_map(inputs, baseline_logits, num_classes=10):
        """
        Section IV-A: Non-targeted reversed JSMA pixel attack.

        Normal JSMA: Corrp = ∂f(x)y/∂xi > 0, Corrn = Σ_{y'≠y} ∂f(x)y'/∂xi < 0
        Reversed (non-targeted): Corr'p = ∂f(x)y/∂xi < 0, Corr'n > 0

        Map = -alpha * beta   when both reversed conditions satisfied
        x̂ = x + Map'

        inputs must already have requires_grad=True and be connected
        to baseline_logits through the computation graph.
        """
        probs = F.softmax(baseline_logits, dim=1)
        batch_size, channels, height, width = inputs.shape

        # Compute per-class Jacobian: ∂f(x)_c / ∂x for all classes
        jacobian = torch.zeros(
            batch_size, num_classes, channels, height, width,
            device=inputs.device
        )

        for target_class in range(num_classes):
            grad_outputs = torch.zeros_like(probs)
            grad_outputs[:, target_class] = 1.0

            grads = torch.autograd.grad(
                outputs=probs,
                inputs=inputs,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]

            if grads is not None:
                jacobian[:, target_class] = grads

        with torch.no_grad():
            true_labels = probs.argmax(dim=1)

            # alpha = ∂f(x)_y / ∂x  (gradient w.r.t. true class)
            alpha = jacobian[torch.arange(batch_size), true_labels]

            # beta = Σ_{y'≠y} ∂f(x)_y' / ∂x  (sum of non-true class gradients)
            beta = torch.zeros_like(alpha)
            for b in range(batch_size):
                y = true_labels[b]
                other_classes = [c for c in range(num_classes) if c != y.item()]
                beta[b] = jacobian[b, other_classes].sum(dim=0)

            # Reversed correlation conditions for non-targeted attack
            corr_p_reversed = (alpha < 0)   # Corr'p: target class gradient negative
            corr_n_reversed = (beta > 0)    # Corr'n: sum of other classes positive
            valid_mask = corr_p_reversed & corr_n_reversed

            # Equation 1: Map = -alpha * beta where conditions satisfied
            saliency_map = torch.where(
                valid_mask,
                -alpha * beta,
                torch.zeros_like(alpha)
            )

            # x̂ = x + Map' — perturbed image, clamped to valid range
            perturbed_x = torch.clamp(inputs.detach() + saliency_map, 0.0, 1.0)

        # Release gradient tracking on inputs after JSMA computation
        inputs.requires_grad_(False)

        return perturbed_x.detach(), saliency_map.detach()

    @staticmethod
    def space_to_depth_downsample(image, saliency_map=None):
        """
        Section IV-B: Reversible spatial downsampling.
        Produces tensor of shape [B, 4*C + 1, H/2, W/2].

        The 4*C channels come from splitting each 2x2 patch into 4 sub-pixels.
        The +1 channel is the saliency map (or zeros if not available).

        For MNIST  (1 channel):  output is [B, 5,  H/2, W/2]
        For CIFAR-10 (3 channels): output is [B, 13, H/2, W/2]
        """
        b, c, h, w = image.shape

        # F.unfold extracts 2×2 non-overlapping patches
        # Output: [B, C*2*2, (H/2)*(W/2)] = [B, C*4, H*W/4]
        downsampled = F.unfold(image, kernel_size=2, stride=2)
        downsampled = downsampled.view(b, c * 4, h // 2, w // 2)

        if saliency_map is None:
            # During inference/evaluation — zero placeholder for saliency channel
            zero_map = torch.zeros(b, 1, h // 2, w // 2, device=image.device)
            return torch.cat([downsampled, zero_map], dim=1)

        # During training — concatenate actual saliency map channel
        # Average across color channels and resize to match downsampled spatial size
        mean_saliency = saliency_map.mean(dim=1, keepdim=True)
        rescaled_map = F.interpolate(
            mean_saliency,
            size=(h // 2, w // 2),
            mode='nearest'
        )
        return torch.cat([downsampled, rescaled_map], dim=1)

    @staticmethod
    def depth_to_space_upsample(downsampled_tensor, original_channels):
        """
        Inverse of space_to_depth_downsample.
        Recovers original spatial resolution for visualization and attack comparison.

        downsampled_tensor: [B, 4*C+1, H/2, W/2]
        original_channels: C (1 for MNIST, 3 for CIFAR-10)

        Returns: [B, C, H, W] recovered in original resolution
        """
        b, _, h_half, w_half = downsampled_tensor.shape

        # Take only the 4*C spatial channels — discard saliency channel
        spatial = downsampled_tensor[:, :original_channels * 4, :, :]

        # Reshape to F.fold input format: [B, C*4, (H/2)*(W/2)]
        spatial_flat = spatial.reshape(b, original_channels * 4, h_half * w_half)

        # F.fold: inverse of F.unfold — recovers original spatial layout
        h_orig = h_half * 2
        w_orig = w_half * 2
        recovered = F.fold(
            spatial_flat,
            output_size=(h_orig, w_orig),
            kernel_size=2,
            stride=2
        )
        return torch.clamp(recovered, 0.0, 1.0)


class PGSLProximalRecoveryBlock(nn.Module):
    """
    Section IV-C: Server-side proximal gradient recovery.
    Solves nuclear norm minimization via Singular Value Thresholding (SVT).

    Objective (Eq 2): min_{x̂, M̄} ||x̂||* − λ||M̄||*
    Relaxed form (Eq 4): F(χ) = µg(χ) + f(χ)
    where f(χ) = (1/2)||x̂ − M̄||² (Eq 5)

    Gradient of f: ∇f(M̄) = M̄ − x̂

    SVT proximal operator for nuclear norm:
    prox_τ(Z) = U·diag(max(σ−τ, 0))·V^T

    mu = 0.55 is the optimal relaxation parameter from paper's
    sensitivity analysis (Figure 3), which minimizes MSE on both
    MIAS and MNIST datasets.
    """

    def __init__(self, mu=0.55, lambda_reg=0.1, iterations=10):
        super(PGSLProximalRecoveryBlock, self).__init__()
        # mu: relaxation parameter — paper's optimal value is 0.55
        self.mu         = mu
        self.lambda_reg = lambda_reg
        self.iterations = iterations

    def forward(self, smashed_activations):
        """
        Applies SVT-based proximal gradient to smashed data.
        smashed_activations: [B, C, H, W] received from client
        Returns: [B, C, H, W] recovered (less invertible) activations
        """
        b, c, h, w = smashed_activations.shape

        # bar_m: approximation of the saliency map component (initialized to zeros)
        bar_m = torch.zeros_like(smashed_activations)

        for _ in range(self.iterations):
            # Gradient of smooth reconstruction loss f(M̄) = (1/2)||x̂ - M̄||²
            # ∇f(M̄) = M̄ - x̂
            # Fix: no factor of 2 because f has (1/2) prefix
            grad_f = bar_m - smashed_activations

            # Gradient descent step: z = M̄ - µ·∇f(M̄)
            z = bar_m - self.mu * grad_f

            # ── SVT: proximal operator for nuclear norm ────────────────────
            # Reshape [B*C, H*W] for per-slice SVD
            z_matrix = z.view(b * c, h * w)

            with torch.no_grad():
                # Singular value decomposition
                u, s, vh = torch.linalg.svd(z_matrix, full_matrices=False)

                # Soft threshold: max(σ − λµ, 0)
                threshold = self.lambda_reg * self.mu
                s_thresholded = torch.clamp(s - threshold, min=0.0)

                # Reconstruct: U · diag(σ_thresh) · V^T
                z_soft = u @ torch.diag_embed(s_thresholded) @ vh
                bar_m = z_soft.view(b, c, h, w)

        # Return: smashed data minus the recovered saliency component
        # This makes the smashed data harder to invert
        return smashed_activations - bar_m


class ConvolutionSumFusion(nn.Module):
    """
    Section IV-D: Convolution-Sum Fusion.
    Fuses attacked stream activations (s_a) with recovered stream (s_r).

    Equation 7: f̂_out = sum(conv(f̂_u, f̂_v), f̂_u)
    Equation 8: f̂_out = (concat * filt + bias) + f̂_u

    Steps: concatenate → conv (reduce dims) → sum with f̂_u
    """

    def __init__(self, channels):
        super(ConvolutionSumFusion, self).__init__()
        # Conv takes 2*channels (concatenated) and reduces back to channels
        self.conv_blend = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, s_a, s_r):
        """
        s_a: attacked stream smashed data [B, C, H, W]
        s_r: recovered stream smashed data [B, C, H, W]
        Returns: fused smashed data [B, C, H, W]
        """
        # Concatenate across channels → [B, 2C, H, W]
        concatenated = torch.cat([s_a, s_r], dim=1)
        # Conv-reduce → [B, C, H, W], then add s_a (skip connection)
        return self.conv_blend(concatenated) + s_a


class AdaptiveWeightedDecisionFusion:
    """
    Section IV-E: Score-based adaptive late fusion.
    Combines predictions from three streams using softmax confidence scores.

    Equations 10-11:
    γ = (W_γ · S^max_r) / denominator
    ρ = (W_ρ · S^max_f) / denominator
    β = (W_β · S^max_a) / denominator

    S_awa = γ·S_r + ρ·S_f + (1−γ−ρ)·S_a

    Initial weights from paper: W_γ=0.5, W_ρ=0.3, W_β=0.2
    Based on individual stream accuracy: S_r > S_f > S_a
    """

    @staticmethod
    def fuse_outputs(out_a, out_r, out_f, w_gamma=0.5, w_rho=0.3, w_beta=0.2):
        """
        out_a, out_r, out_f: raw logits from three streams [B, num_classes]
        Returns: fused logit scores [B, num_classes]
        """
        with torch.no_grad():
            # S^max for each stream: max softmax probability across classes
            s_a_max = F.softmax(out_a, dim=1).max(dim=1, keepdim=True)[0]
            s_r_max = F.softmax(out_r, dim=1).max(dim=1, keepdim=True)[0]
            s_f_max = F.softmax(out_f, dim=1).max(dim=1, keepdim=True)[0]

            denominator = (
                w_gamma * s_r_max +
                w_rho   * s_f_max +
                w_beta  * s_a_max +
                1e-8
            )

            # Adaptive weights per sample
            gamma = (w_gamma * s_r_max) / denominator
            rho   = (w_rho   * s_f_max) / denominator

        # Equation 9: S_awa = γ·S_r + ρ·S_f + (1-γ-ρ)·S_a
        return gamma * out_r + rho * out_f + (1.0 - gamma - rho) * out_a