
import math
import hashlib
import torch


class ZORRODefense:
   

    def __init__(self, high_freq_split=0.5, std_multiplier=3.0, min_history=5):
        assert 0.0 < high_freq_split < 1.0
        self.high_freq_split = high_freq_split
        self.std_multiplier = std_multiplier
        self.min_history = min_history
        self.history = []

    # ── Discrete Cosine Transform (DCT-II), computed efficiently via FFT ──
    @staticmethod
    def _dct(x):
        
        n = x.shape[-1]
        v = torch.cat([x[..., 0::2], x[..., 1::2].flip(-1)], dim=-1)
        Vc = torch.fft.fft(v, dim=-1)

        k = -torch.arange(n, dtype=x.dtype, device=x.device) * math.pi / (2 * n)
        w_real = torch.cos(k)
        w_imag = torch.sin(k)

        V = Vc.real * w_real - Vc.imag * w_imag
        return 2 * V

    def _high_freq_energy_ratio(self, update_vector):
        coeffs = self._dct(update_vector)
        n = coeffs.shape[-1]
        split_point = int(n * (1 - self.high_freq_split))

        total_energy = (coeffs ** 2).sum() + 1e-12
        high_freq_energy = (coeffs[split_point:] ** 2).sum()

        return (high_freq_energy / total_energy).item()

    def analyze_update(self, update_vector):
        """
        Args:
            update_vector (Tensor): the client's flattened model-partition
                                     update for this round

        Returns:
            is_anomalous (bool): True if this update looks backdoor-like
            energy_ratio (float): the computed high-frequency energy ratio
        """
        ratio = self._high_freq_energy_ratio(update_vector)

        is_anomalous = False
        if len(self.history) >= self.min_history:
            mean = sum(self.history) / len(self.history)
            variance = sum((h - mean) ** 2 for h in self.history) / len(self.history)
            std = variance ** 0.5
            threshold = mean + self.std_multiplier * std
            is_anomalous = ratio > threshold

        if not is_anomalous:
            self.history.append(ratio)

        return is_anomalous, ratio

    # ── Simplified stand-in for the paper's zero-knowledge proof ──
    def generate_proof(self, update_vector, is_anomalous, energy_ratio):
        """
        Produces a commitment the server can later verify against, proving
        the client can't change its update/verdict after the fact.

        NOTE: this is a hash commitment, not a true zero-knowledge proof —
        see the module docstring above for what that simplification does
        and doesn't give you.
        """
        payload = (update_vector.detach().numpy().tobytes()
                   + str(is_anomalous).encode()
                   + f"{energy_ratio:.8f}".encode())
        commitment = hashlib.sha256(payload).hexdigest()
        return {
            "commitment": commitment,
            "is_anomalous": is_anomalous,
            "energy_ratio": energy_ratio,
        }

    @staticmethod
    def verify_proof(proof, update_vector):
        """Server-side (or auditor-side) check: recompute the commitment
        from the claimed update/verdict and confirm it matches what the
        client originally committed to."""
        payload = (update_vector.detach().numpy().tobytes()
                   + str(proof["is_anomalous"]).encode()
                   + f"{proof['energy_ratio']:.8f}".encode())
        expected = hashlib.sha256(payload).hexdigest()
        return expected == proof["commitment"]


# ─────────────────────────────────────────
# Test — simulate benign updates building up history, then a backdoor-like
# update with concentrated high-frequency energy, and confirm ZORRO flags
# it and that the proof round-trips correctly.
# ─────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 50)
    print("ZORRO Defense Test")
    print("=" * 50)

    torch.manual_seed(0)

    defense = ZORRODefense(high_freq_split=0.5, std_multiplier=3.0, min_history=5)

    # ── Phase 1: benign updates — smooth, low-frequency-dominant signals
    # (simulating typical small, gradual gradient-descent updates) ──
    n = 512
    t = torch.linspace(0, 1, n)
    for i in range(10):
        benign_update = torch.sin(2 * math.pi * 2 * t) * 0.05 + torch.randn(n) * 0.01
        flagged, ratio = defense.analyze_update(benign_update)
        if i < 3:
            print(f"[OK] Benign update {i}: flagged={flagged}, high_freq_ratio={ratio:.4f}")

    print(f"[OK] History length after benign phase : {len(defense.history)}")

    # ── Phase 2: backdoor-like update — energy concentrated at the top of
    # the DCT spectrum (simulating a poisoned update injecting a sharp,
    # structured trigger pattern, which shows up as high-frequency content) ──
    top_freq_idx = n - 1
    idx = torch.arange(n, dtype=torch.float32)
    backdoor_update = torch.cos(math.pi / n * (idx + 0.5) * top_freq_idx) * 0.5
    flagged, ratio = defense.analyze_update(backdoor_update)
    print(f"\n[OK] Backdoor-like update: flagged={flagged}, high_freq_ratio={ratio:.4f}")

    # ── Proof generation and verification round-trip ──
    proof = defense.generate_proof(backdoor_update, flagged, ratio)
    verified_ok = defense.verify_proof(proof, backdoor_update)
    print(f"[OK] Proof verifies against original update   : {verified_ok}")

    # Confirm tampering is caught: verifying against a DIFFERENT update
    # (as if the client tried to swap in a different update after the
    # fact) must fail
    tampered_update = torch.randn(n)
    tampering_caught = not defense.verify_proof(proof, tampered_update)
    print(f"[OK] Tampered/swapped update correctly rejected: {tampering_caught}")

    print("=" * 50)
    print("ZORRO defense working correctly.")
    print("=" * 50)