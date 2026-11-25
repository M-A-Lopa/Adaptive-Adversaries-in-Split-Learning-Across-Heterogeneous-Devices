import math
import torch
import torch.nn as nn


def clip_k(v, K, C):
    N = v.shape[1]
    K = max(1, min(int(K), N))
    top_vals, top_idx = torch.topk(v, K, dim=1)
    out = torch.zeros_like(v)
    out.scatter_(1, top_idx, top_vals)
    return out.clamp(0.0, C)


def rr_probability(ratio, eps_p, K):
    e = math.exp(eps_p / K)
    return 0.5 + ratio * (e / (1.0 + e) - 0.5)


def sample_laplace(shape, scale, device):
    u = torch.rand(shape, device=device) - 0.5
    return -scale * torch.sign(u) * torch.log1p(-2.0 * u.abs().clamp(max=0.5 - 1e-7))


class DynamicImportance:

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
        w = self.U / (self.U.mean() + 1e-12)
        return w.clamp(min=w_min)


class R3eLUMechanism:

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
        return eps / 2.0, eps / 2.0

    def laplace_scale(self, C):
        _, eps_l = self._budgets()
        return 2.0 * self.K * C / eps_l

    @torch.no_grad()
    def forward_perturb(self, v):
        shape = v.shape
        flat = v.reshape(shape[0], -1)
        eps_p, _ = self._budgets()

        v_hat = clip_k(flat, self.K, self.C)
        if self.dynamic_budget and self.importance.ready():
            ratio = self.importance.ratio().unsqueeze(0).expand_as(v_hat)
        else:
            ratio = v_hat / (v_hat.abs().amax(dim=1, keepdim=True) + 1e-12)
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
        shape = delta.shape
        flat = delta.reshape(shape[0], -1)
        eps_p, _ = self._budgets()

        abs_hat = clip_k(flat.abs(), self.K, self.C_backward)
        ratio = abs_hat / (abs_hat.amax(dim=1, keepdim=True) + 1e-12)
        p = rr_probability(ratio, eps_p, self.K)

        keep = torch.rand_like(p) < p
        signed = torch.where(keep, torch.sign(flat) * abs_hat, torch.zeros_like(abs_hat))
        out = signed + sample_laplace(signed.shape, self.laplace_scale(self.C_backward), delta.device)
        return out.reshape(shape)

    def end_iteration(self):
        self.step += 1

    def accountant(self, total_steps, sampling_ratio, delta=1e-5):
        ge = sampling_ratio * self.epsilon
        return ge * math.sqrt(2 * total_steps * math.log(1 / delta)) + ge * total_steps * (math.exp(ge) - 1)

    def __repr__(self):
        return (f"R3eLUMechanism(eps={self.epsilon}, eps_p=eps_l={self.epsilon/2}, K={self.K}/{self.N}, "
                f"C={self.C}, lap_scale={self.laplace_scale(self.C):.1f}, dynamic={self.dynamic_budget}) "
                f"[Mao et al. 2023]")


class _R3eLUFunction(torch.autograd.Function):

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

    def __init__(self, mechanism):
        super().__init__()
        self.mech = mechanism

    def forward(self, v):
        return _R3eLUFunction.apply(v, self.mech)


class R3eLUDefense:

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