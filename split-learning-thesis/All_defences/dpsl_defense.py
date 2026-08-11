import torch


class DPSLDefense:

    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=32.0):
        assert epsilon > 0, "epsilon must be positive"
        assert 0 < delta < 1, "delta must be in (0, 1)"

        self.epsilon   = epsilon
        self.delta     = delta
        self.clip_norm = clip_norm

        self.sigma = self._compute_sigma()

    def _compute_sigma(self):
        import math
        return self.clip_norm * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def _clip(self, smashed_data):
        batch_size = smashed_data.shape[0]
        flat = smashed_data.view(batch_size, -1)
        norms = flat.norm(p=2, dim=1, keepdim=True) 

        scale = torch.clamp(self.clip_norm / (norms + 1e-8), max=1.0)
        clipped_flat = flat * scale
        return clipped_flat.view_as(smashed_data)

    def protect(self, smashed_data):
        clipped = self._clip(smashed_data)
        noise = torch.randn_like(clipped) * self.sigma
        return clipped + noise

    def __repr__(self):
        return (f"DPSLDefense(epsilon={self.epsilon}, delta={self.delta}, "
                f"clip_norm={self.clip_norm}, sigma={self.sigma:.4f})")


if __name__ == "__main__":

    torch.manual_seed(0)

    dummy_smashed = torch.randn(4, 32, 8, 8)  

    for eps in [0.1, 1.0, 10.0]:
        dpsl = DPSLDefense(epsilon=eps, delta=1e-5, clip_norm=1.0)
        protected = dpsl.protect(dummy_smashed)
        diff = (protected - dummy_smashed).norm().item()
        print(f"{dpsl}")
        print(f"  Input shape : {tuple(dummy_smashed.shape)}")
        print(f"  Output shape: {tuple(protected.shape)}")
        print(f"  L2 distance introduced by defense: {diff:.4f}\n")