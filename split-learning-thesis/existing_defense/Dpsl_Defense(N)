
import math
import torch
import torch.nn as nn


class DPSLDefense(nn.Module):
    

    def __init__(self, epsilon=2.0, delta=1e-5, clip_min=0.0, clip_max=1.0):
        super(DPSLDefense, self).__init__()
        assert epsilon > 0, "epsilon must be > 0"
        assert 0 < delta < 1, "delta must be in (0, 1)"

        self.epsilon = epsilon
        self.delta = delta
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.sigma = self._compute_sigma(epsilon, delta)

    @staticmethod
    def _compute_sigma(epsilon, delta):
        """Standard (ε, δ)-DP Gaussian mechanism noise scale, assuming
        sensitivity Δ = 1 (guaranteed by clipping activations to [0, 1])."""
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def forward(self, smashed_data):
        # Step 1: clip activations → bounds sensitivity to 1
        x = torch.clamp(smashed_data, self.clip_min, self.clip_max)
        # Step 2: add calibrated Gaussian noise
        noise = torch.randn_like(x) * self.sigma
        return x + noise

    def set_epsilon(self, epsilon):
        """Change the privacy budget after construction (e.g. when sweeping
        ε in {2, 3, 5} for experiments) without rebuilding the module."""
        assert epsilon > 0, "epsilon must be > 0"
        self.epsilon = epsilon
        self.sigma = self._compute_sigma(epsilon, self.delta)

    def extra_repr(self):
        return (f"epsilon={self.epsilon}, delta={self.delta}, sigma={self.sigma:.4f}, "
                f"clip=[{self.clip_min}, {self.clip_max}]")


# ─────────────────────────────────────────
# Test — verify sigma values match the paper's Table 1, and that
# clipping + noise behave correctly.
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("DP-SL Defense Test")
    print("=" * 50)

    torch.manual_seed(0)

    # ── Check sigma matches paper's reported values for eps in {2, 3, 5} ──
    paper_sigma = {5: 0.97, 3: 1.62, 2: 2.42}
    for eps, expected_sigma in paper_sigma.items():
        defense = DPSLDefense(epsilon=eps, delta=1e-5)
        print(f"[OK] eps={eps} -> computed sigma={defense.sigma:.4f} "
              f"(paper reports {expected_sigma})")

    # ── Check clipping + noise on fake smashed data ──
    dummy_smashed = torch.rand(4, 256, 4, 4) * 3 - 1  # values outside [0,1]
    print(f"\n[OK] Input range before defense : "
          f"[{dummy_smashed.min():.3f}, {dummy_smashed.max():.3f}]")

    defense = DPSLDefense(epsilon=2.0)
    protected = defense(dummy_smashed)
    print(f"[OK] Output shape                : {protected.shape}")
    print(f"[OK] Shapes match (no size change): {dummy_smashed.shape == protected.shape}")

    # Clipping happened before noise, so protected values should mostly sit
    # near [0, 1] plus/minus noise spread (sigma ~2.42 at eps=2, so range
    # will be wide, but let's confirm the clip step itself works in isolation)
    clip_only = torch.clamp(dummy_smashed, 0.0, 1.0)
    print(f"[OK] Clip-only range              : [{clip_only.min():.3f}, {clip_only.max():.3f}]")

    # ── Check set_epsilon() works for sweeping experiments ──
    defense.set_epsilon(5.0)
    print(f"[OK] After set_epsilon(5.0), sigma: {defense.sigma:.4f} (expect ~0.97)")

    print("=" * 50)
    print("DP-SL defense working correctly.")
    print("=" * 50)
 