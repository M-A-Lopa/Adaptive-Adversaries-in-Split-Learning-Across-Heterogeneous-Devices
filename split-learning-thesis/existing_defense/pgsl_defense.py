import torch
import torch.nn as nn
import torch.nn.functional as F


class PGSLDefenseModules:
    @staticmethod
    def generate_faithful_saliency_map(inputs, baseline_logits, num_classes=10):
        probs = F.softmax(baseline_logits, dim=1)
        batch_size, channels, height, width = inputs.shape

        jacobian = torch.zeros(batch_size, num_classes, channels, height, width, device=inputs.device)

        for target_class in range(num_classes):
            grad_outputs = torch.zeros_like(probs)
            grad_outputs[:, target_class] = 1.0

            grads = torch.autograd.grad(outputs=probs, inputs=inputs, grad_outputs=grad_outputs, retain_graph=True, create_graph=False, allow_unused=True)[0]

            if grads is not None:
                jacobian[:, target_class] = grads

        with torch.no_grad():
            true_labels = probs.argmax(dim=1)
            alpha = jacobian[torch.arange(batch_size), true_labels]
            beta = torch.zeros_like(alpha)
            
            for b in range(batch_size):
                y = true_labels[b]
                other_classes = [c for c in range(num_classes) if c != y.item()]
                beta[b] = jacobian[b, other_classes].sum(dim=0)

            corr_p_reversed = (alpha < 0)  
            corr_n_reversed = (beta > 0)  
            valid_mask = corr_p_reversed & corr_n_reversed

            saliency_map = torch.where(valid_mask, -alpha * beta, torch.zeros_like(alpha))

            perturbed_x = torch.clamp(inputs.detach() + saliency_map, 0.0, 1.0)

        inputs.requires_grad_(False)

        return perturbed_x.detach(), saliency_map.detach()

    @staticmethod
    def space_to_depth_downsample(image, saliency_map=None):
        b, c, h, w = image.shape

        downsampled = F.unfold(image, kernel_size=2, stride=2)
        downsampled = downsampled.view(b, c * 4, h // 2, w // 2)

        if saliency_map is None:
            zero_map = torch.zeros(b, 1, h // 2, w // 2, device=image.device)
            return torch.cat([downsampled, zero_map], dim=1)

        mean_saliency = saliency_map.mean(dim=1, keepdim=True)
        rescaled_map = F.interpolate(mean_saliency, size=(h // 2, w // 2), mode='nearest')
        return torch.cat([downsampled, rescaled_map], dim=1)

    @staticmethod
    def depth_to_space_upsample(downsampled_tensor, original_channels):
        b, _, h_half, w_half = downsampled_tensor.shape

        spatial = downsampled_tensor[:, :original_channels * 4, :, :]

        spatial_flat = spatial.reshape(b, original_channels * 4, h_half * w_half)

        h_orig = h_half * 2
        w_orig = w_half * 2
        recovered = F.fold(spatial_flat, output_size=(h_orig, w_orig), kernel_size=2, stride=2)
        return torch.clamp(recovered, 0.0, 1.0)


class PGSLProximalRecoveryBlock(nn.Module):
    def __init__(self, mu=0.55, lambda_reg=0.1, iterations=10):
        super(PGSLProximalRecoveryBlock, self).__init__()
        self.mu         = mu
        self.lambda_reg = lambda_reg
        self.iterations = iterations

    def forward(self, smashed_activations):
        b, c, h, w = smashed_activations.shape

        bar_m = torch.zeros_like(smashed_activations)

        for _ in range(self.iterations):
            grad_f = bar_m - smashed_activations

            z = bar_m - self.mu * grad_f

            z_matrix = z.view(b * c, h * w)

            with torch.no_grad():
                u, s, vh = torch.linalg.svd(z_matrix, full_matrices=False)

                threshold = self.lambda_reg * self.mu
                s_thresholded = torch.clamp(s - threshold, min=0.0)

                z_soft = u @ torch.diag_embed(s_thresholded) @ vh
                bar_m = z_soft.view(b, c, h, w)

        return smashed_activations - bar_m


class ConvolutionSumFusion(nn.Module):
    def __init__(self, channels):
        super(ConvolutionSumFusion, self).__init__()
        self.conv_blend = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())

    def forward(self, s_a, s_r):
        concatenated = torch.cat([s_a, s_r], dim=1)
        
        return self.conv_blend(concatenated) + s_a


class AdaptiveWeightedDecisionFusion:
    @staticmethod
    def fuse_outputs(out_a, out_r, out_f, w_gamma=0.5, w_rho=0.3, w_beta=0.2):

        with torch.no_grad():
            s_a_max = F.softmax(out_a, dim=1).max(dim=1, keepdim=True)[0]
            s_r_max = F.softmax(out_r, dim=1).max(dim=1, keepdim=True)[0]
            s_f_max = F.softmax(out_f, dim=1).max(dim=1, keepdim=True)[0]

            denominator = (w_gamma * s_r_max + w_rho   * s_f_max + w_beta  * s_a_max + 1e-8)

            gamma = (w_gamma * s_r_max) / denominator
            rho   = (w_rho   * s_f_max) / denominator

        return gamma * out_r + rho * out_f + (1.0 - gamma - rho) * out_a