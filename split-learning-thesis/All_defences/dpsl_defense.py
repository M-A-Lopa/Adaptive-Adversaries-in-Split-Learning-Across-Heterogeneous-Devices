# dpsl_defense.py
# Differentially Private Split Learning (DP-SL) Defense
# Based on: Thapa et al., "Enhancing accuracy-privacy trade-off in
# differentially private split learning" (IEEE TIFS, 2024, arXiv:2310.14434)
#
# Core idea: before smashed data leaves the client and reaches the
# server (or an attacker intercepting it), we:
#   1. CLIP each sample's smashed data to a bounded L2 norm C
#      (this bounds "sensitivity" — how much one sample can influence output)
#   2. Add calibrated Gaussian noise, scaled so the whole operation
#      satisfies (epsilon, delta)-Differential Privacy
#
# This directly targets model inversion attacks (like the White-Box  Inversion Attack)
# because the attacker's optimization now chases a noisy target instead
# of the true smashed data, degrading reconstruction quality.

import torch


class DPSLDefense:
    """
    Differentially Private defense for Split Learning smashed data.

    Usage (matches the defense_fn signature expected by
    SplitLearningTrainer.evaluate_with_defense and
    UnsplitAttacker.run_attack_with_defense):

        dpsl = DPSLDefense(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        protected_smashed = dpsl.protect(smashed_data)
    """

    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=32.0):
        """
        epsilon   : privacy budget. LOWER epsilon = MORE privacy = MORE noise
                    (typical thesis sweep: 0.1, 0.5, 1.0, 5.0, 10.0)
        delta     : probability of privacy failure, standard choice 1e-5
        clip_norm : C, the L2 norm bound each sample's smashed data is
                    clipped to before noise is added
        """
        assert epsilon > 0, "epsilon must be positive"
        assert 0 < delta < 1, "delta must be in (0, 1)"

        self.epsilon   = epsilon
        self.delta     = delta
        self.clip_norm = clip_norm

        # Gaussian mechanism noise scale (Dwork & Roth, 2014, Theorem 3.22):
        # sigma = C * sqrt(2 * ln(1.25 / delta)) / epsilon
        self.sigma = self._compute_sigma()

    def _compute_sigma(self):
        import math
        return self.clip_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def _clip(self, smashed_data):
        """
        Clips each sample in the batch to L2 norm <= clip_norm.
        smashed_data: [B, C, H, W]
        """
        batch_size = smashed_data.shape[0]
        flat = smashed_data.view(batch_size, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True)  # [B, 1]

        scale = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
        clipped_flat = flat * scale
        return clipped_flat.view_as(smashed_data)

    def protect(self, smashed_data):
        """
        Applies clip + Gaussian noise to smashed data.

        smashed_data: [B, C, H, W] tensor (client's intermediate activations)
        returns: perturbed tensor of the SAME shape — safe to feed into
                 the server model or an attacker's reconstruction routine.
        """
        clipped = self._clip(smashed_data)
        noise = torch.randn_like(clipped) * self.sigma
        return clipped + noise

    def __repr__(self):
        return (f"DPSLDefense(epsilon={self.epsilon}, delta={self.delta}, "
                f"clip_norm={self.clip_norm}, sigma={self.sigma:.4f})")


if __name__ == "__main__":
    # Quick self-test — run this file directly to sanity check the math
    # and shapes before wiring it into the full pipeline.
    torch.manual_seed(0)

    dummy_smashed = torch.randn(4, 32, 8, 8)  # simulate a CIFAR-10 smashed batch

    for eps in [0.1, 1.0, 10.0]:
        dpsl = DPSLDefense(epsilon=eps, delta=1e-5, clip_norm=1.0)
        protected = dpsl.protect(dummy_smashed)
        diff = (protected - dummy_smashed).norm().item()
        print(f"{dpsl}")
        print(f"  Input shape : {tuple(dummy_smashed.shape)}")
        print(f"  Output shape: {tuple(protected.shape)}")
        print(f"  L2 distance introduced by defense: {diff:.4f}\n")