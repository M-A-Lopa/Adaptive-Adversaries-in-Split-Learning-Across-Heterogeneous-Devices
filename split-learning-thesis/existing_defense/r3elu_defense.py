"""
R3eLU — Randomized-Response ReLU privacy tunnel for split learning.

Paper : Y. Mao, Z. Xin, Z. Li, J. Hong, Q. Yang, S. Zhong,
        "Secure Split Learning against Property Inference, Data Reconstruction,
        and Feature Space Hijacking Attacks", ESORICS 2023 / arXiv:2304.09515.

No official code was released by the authors. This file is a line-by-line
implementation of the paper's Algorithm 1 (R3eLU-forward), Algorithm 2
(R3eLU-backward), Eq.(4)/(6) (randomized-response probabilities),
Eq.(8)/(9)/(10) (dynamic feature-importance budget allocation) and
Corollary 4/5 (strong-composition privacy accounting).

Paper defaults (Sec. 5): eps_p = eps_l = eps/2, K = N/2, C = 10,
initial importance = 0 for all features.

Roles:
    guest (client) calls  R3eLU.forward   before sending smashed data
    host  (server) calls  backward_perturb before sending the cut-layer grad
"""
import math
import torch
import torch.nn as nn


def clip_k(v, K, C):
    """
    ClipK(v, C, K, N): keep the K largest entries of every row, zero the rest,
    then clip each kept value into [0, C]  (Corollary 1: 0 <= v_hat_i <= C).
    v : [B, N]
    """
    N = v.shape[1]
    K = max(1, min(int(K), N))
    top_vals, top_idx = torch.topk(v, K, dim=1)
    out = torch.zeros_like(v)
    out.scatter_(1, top_idx, top_vals)
    return out.clamp(0.0, C)


def rr_probability(ratio, eps_p, K):
    """Eq.(4)/(6)/(10):  p_i = 1/2 + ratio_i * ( e^{eps_p/K} / (1+e^{eps_p/K}) - 1/2 )."""
    e = math.exp(eps_p / K)
    return 0.5 + ratio * (e / (1.0 + e) - 0.5)


def sample_laplace(shape, scale, device):
    """Lap(0, scale). `scale` may be a float or a broadcastable tensor."""
    u = torch.rand(shape, device=device) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp(max=0.5 - 1e-7))


class DynamicImportance:
    """
    Sec. 3.4. First-order Taylor importance of every cut-layer neuron
    (Eq. 8, Molchanov et al.) accumulated as a running mean over iterations (Eq. 9).
    As stated in the paper, the O(N_u) cost comes from the product of the cut-layer
    gradient and the neuron values, so  I_j = (grad_j * a_j)^2  summed over the batch.
    Only the gradient the guest actually RECEIVES is used (post-processing, no extra leakage).
    """

    def __init__(self, num_features, device):
        self.U = torch.zeros(num_features, device=device)
        self.t = 0

    @torch.no_grad()
    def update(self, activations, grad):
        a = activations.reshape(activations.shape[0], -1)
        g = grad.reshape(grad.shape[0], -1)
        cur = ((a * g) ** 2).sum(dim=0)
        self.t += 1
        self.U = (self.U * (self.t - 1) + cur) / self.t

    def ready(self):
        return self.t > 0 and float(self.U.max()) > 0.0

    def ratio(self):
        return self.U / (self.U.max() + 1e-12)

    def laplace_weight(self, w_min=0.1):
        # Eq.(16): feature j receives budget share  eps_l * U_j / sum(U).
        # Normalised so the mean weight is 1 -> reduces exactly to Algorithm 1 when U is uniform.
        w = self.U / (self.U.mean() + 1e-12)
        return w.clamp(min=w_min)


class R3eLUMechanism:
    """
    Holds the privacy parameters and implements both procedures.
        epsilon        : total per-step budget eps = eps_p + eps_l  (paper splits it 50/50)
        K              : top-K kept values (paper: half the number of features)
        C              : clipping constant (paper: 10)
        dynamic_budget : Sec. 3.4 importance-based allocation (paper: enabled)
        iteration_decay: eps_i = eps_T / 2^i per iteration (Sec. 3.4, [Du et al.]).
                         Off by default: after ~20 iterations the Laplace scale becomes astronomically
                         large, so the paper's reported accuracies are only reachable without it.
    """

    def __init__(self, num_features, epsilon=1.0, K=None, C=10.0, C_backward=None, dynamic_budget=True,
                 iteration_decay=False, protect_forward=True, protect_backward=True, device='cpu'):
        assert epsilon > 0
        self.N = int(num_features)
        self.epsilon = float(epsilon)
        self.K = int(K) if K is not None else max(1, self.N // 2)
        self.C = float(C)
        self.C_backward = float(C_backward) if C_backward is not None else float(C)
        self.dynamic_budget = dynamic_budget
        self.iteration_decay = iteration_decay
        self.protect_forward = protect_forward
        self.protect_backward = protect_backward
        self.importance = DynamicImportance(self.N, device)
        self.step = 0

    def _budgets(self):
        eps = self.epsilon / (2 ** (self.step + 1)) if self.iteration_decay else self.epsilon
        return eps / 2.0, eps / 2.0          # eps_p, eps_l

    def laplace_scale(self, C):
        _, eps_l = self._budgets()
        return 2.0 * self.K * C / eps_l      # sensitivity 2KC (Corollary 1) / eps_l

    @torch.no_grad()
    def forward_perturb(self, v):
        """Algorithm 1 (R3eLU-forward). v: raw cut-layer pre-activation, any shape [B, ...]."""
        shape = v.shape
        flat = v.reshape(shape[0], -1)
        eps_p, _ = self._budgets()

        v_hat = clip_k(flat, self.K, self.C)
        if self.dynamic_budget and self.importance.ready():
            ratio = self.importance.ratio().unsqueeze(0).expand_as(v_hat)                  # Eq.(10)
        else:
            ratio = v_hat / (v_hat.abs().amax(dim=1, keepdim=True) + 1e-12)                # Eq.(4)
        p = rr_probability(ratio, eps_p, self.K)

        scale = self.laplace_scale(self.C)
        if self.dynamic_budget and self.importance.ready():
            scale = scale / self.importance.laplace_weight().unsqueeze(0)

        activate = torch.rand_like(p) < p
        noisy = torch.clamp(v_hat + sample_laplace(v_hat.shape, scale, v.device), min=0.0)
        out = torch.where(activate, noisy, torch.zeros_like(noisy))
        return out.reshape(shape)

    @torch.no_grad()
    def backward_perturb(self, delta):
        """Algorithm 2 (R3eLU-backward). delta: partial loss w.r.t. the cut layer, run by the HOST."""
        shape = delta.shape
        flat = delta.reshape(shape[0], -1)
        eps_p, _ = self._budgets()

        abs_hat = clip_k(flat.abs(), self.K, self.C_backward)
        ratio = abs_hat / (abs_hat.amax(dim=1, keepdim=True) + 1e-12)                       # Eq.(6)
        p = rr_probability(ratio, eps_p, self.K)

        keep = torch.rand_like(p) < p
        signed = torch.where(keep, torch.sign(flat) * abs_hat, torch.zeros_like(abs_hat))
        out = signed + sample_laplace(signed.shape, self.laplace_scale(self.C_backward), delta.device)
        return out.reshape(shape)

    def end_iteration(self):
        self.step += 1

    def accountant(self, total_steps, sampling_ratio, delta=1e-5):
        """Corollary 4/5 (strong composition): eps_total = g*e*sqrt(2T ln(1/d)) + g*e*T*(e^{g*e}-1)."""
        ge = sampling_ratio * self.epsilon
        return ge * math.sqrt(2 * total_steps * math.log(1 / delta)) + ge * total_steps * (math.exp(ge) - 1)

    def __repr__(self):
        return (f"R3eLUMechanism(eps={self.epsilon}, eps_p=eps_l={self.epsilon/2}, K={self.K}/{self.N}, "
                f"C={self.C}, lap_scale={self.laplace_scale(self.C):.1f}, dynamic={self.dynamic_budget}) "
                f"[Mao et al. 2023]")


class _R3eLUFunction(torch.autograd.Function):
    """Forward: guest-side R3eLU-forward. Backward: pass-through (the host already perturbed the grad)."""

    @staticmethod
    def forward(ctx, v, mech):
        out = mech.forward_perturb(v) if mech.protect_forward else torch.relu(v)
        ctx.mech = mech
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        ctx.mech.importance.update(out, grad_out)
        if not ctx.mech.protect_backward:
            grad_out = grad_out * (out > 0).float()
        return grad_out, None


class R3eLU(nn.Module):
    """Drop-in cut-layer activation. Train mode -> randomized response; eval mode -> same (DP at inference too)."""

    def __init__(self, mechanism):
        super().__init__()
        self.mech = mechanism

    def forward(self, v):
        return _R3eLUFunction.apply(v, self.mech)


class R3eLUDefense:
    """
    Same interface as DPSLDefense so it works with SplitLearningTrainer.evaluate_with_defense()
    and with the attack runners:  protected = defense.protect(smashed)
    """

    def __init__(self, num_features, epsilon=1.0, **kw):
        self.mech = R3eLUMechanism(num_features, epsilon=epsilon, **kw)

    def protect(self, smashed):
        return self.mech.forward_perturb(smashed)

    def __repr__(self):
        return repr(self.mech)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.relu(torch.randn(4, 32, 7, 7))
    for eps in [0.1, 1.0, 4.0]:
        d = R3eLUDefense(num_features=32 * 7 * 7, epsilon=eps)
        y = d.protect(x)
        print(d)
        print(f"  active={(y > 0).float().mean():.3f}  range=[{y.min():.1f}, {y.max():.1f}]")
    g = torch.randn(4, 32, 7, 7) * 1e-3
    print("  backward out std:", d.mech.backward_perturb(g).std().item())
