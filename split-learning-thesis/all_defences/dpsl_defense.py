

import torch
import math


class DPSLDefense:
    """
    Paper-exact DP-SL defense.

    Usage:
        dpsl = DPSLDefense(epsilon=2.0, delta=1e-5)
        protected_smashed = dpsl.protect(smashed_data)
    """

    def __init__(self, epsilon=2.0, delta=1e-5, sensitivity=1.0):
        """
        epsilon     : privacy budget. Paper tests 2, 3, 4, 5, 10.
                      Lower = more privacy = more noise.
        delta       : failure probability, paper doesn't specify exactly
                      but 1e-5 is the standard DP convention.
        sensitivity : fixed at 1.0 because we clamp outputs to [0,1]
                      (this IS the paper's definition of "1-sensitive").
        """
        assert epsilon > 0, "epsilon must be positive"
        assert 0 < delta < 1, "delta must be in (0, 1)"

        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()

    def _compute_sigma(self):

        return math.sqrt(2 * (self.sensitivity ** 2) * math.log(1.25 / self.delta)) / self.epsilon

    def protect(self, smashed_data):
        """
        Applies the paper's exact mechanism:
        clamp to [0,1], then add Gaussian noise N(0, sigma^2) element-wise.

        smashed_data: [B, C, H, W] tensor
        returns: perturbed tensor of the SAME shape
        """
        clamped = torch.clamp(smashed_data, 0.0, 1.0)
        noise = torch.randn_like(clamped) * self.sigma
        return clamped + noise

    def __repr__(self):
        return (f"DPSLDefense(epsilon={self.epsilon}, delta={self.delta}, "
                f"sensitivity={self.sensitivity}, sigma={self.sigma:.4f})  "
                f"[paper-exact: Pham et al. 2024]")


if __name__ == "__main__":
    torch.manual_seed(0)
    dummy_smashed = torch.randn(4, 32, 7, 7) * 5  # simulate realistic MNIST smashed data

    for eps in [2, 3, 5, 10]:
        dpsl = DPSLDefense(epsilon=eps, delta=1e-5)
        protected = dpsl.protect(dummy_smashed)
        print(f"{dpsl}")
        print(f"  Output range: [{protected.min():.2f}, {protected.max():.2f}]\n")