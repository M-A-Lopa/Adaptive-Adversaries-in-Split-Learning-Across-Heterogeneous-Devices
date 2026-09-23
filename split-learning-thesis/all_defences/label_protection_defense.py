import math
import time
import torch
import torch.nn as nn

from all_defences.marvell_solver import solve_isotropic_covariance, symKL_objective


OFFICIAL_NAME_MAP = {
    'identity': 'no_noise',
    'no_noise': 'no_noise',
    'sumKL': 'marvell',
    'marvell': 'marvell',
    'expectation': 'max_norm',
    'max_norm': 'max_norm',
    'white_gaussian': 'iso',
    'iso': 'iso',
    'perp': 'perp',
}


def no_noise(g):
    return g


def max_norm_perturb(g):
    g_original_shape = g.shape
    g = g.reshape(g_original_shape[0], -1)

    g_norm = torch.norm(g, dim=1, keepdim=True).reshape(-1, 1)
    max_norm = torch.max(g_norm)
    stds = torch.sqrt(torch.clamp(max_norm ** 2 / (g_norm ** 2 + 1e-32) - 1.0, min=0.0))
    standard_gaussian_noise = torch.randn(g.shape[0], 1, device=g.device, dtype=g.dtype)
    gaussian_noise = standard_gaussian_noise * stds
    res = g * (1 + gaussian_noise)

    return res.reshape(g_original_shape)


def iso_gaussian_perturb(g, ratio=1.0):
    g_original_shape = g.shape
    g = g.reshape(g_original_shape[0], -1)

    g_norm = torch.norm(g, dim=1, keepdim=False)
    max_norm = torch.max(g_norm)
    std = ratio * max_norm / math.sqrt(float(g.shape[1]))
    gaussian_noise = torch.randn_like(g) * std

    return (g + gaussian_noise).reshape(g_original_shape)


def perp_perturb(g, lower=1.0, upper=10.0):
    g_original_shape = g.shape
    g = g.reshape(g_original_shape[0], -1)

    g_norm = torch.norm(g, dim=1, keepdim=True)
    max_norm = torch.max(g_norm)
    std_gaussian_noise = torch.randn_like(g)
    inner_product = torch.sum(std_gaussian_noise * g, dim=1, keepdim=True)
    init_perp = std_gaussian_noise - (inner_product / (g_norm ** 2 + 1e-16)) * g
    unit_perp = nn.functional.normalize(init_perp, p=2, dim=1, eps=1e-12)
    norm_to_align = torch.empty_like(g_norm).uniform_(float(lower * max_norm), float(upper * max_norm))
    perp = torch.sqrt(norm_to_align ** 2 - g_norm ** 2 + 1e-8) * unit_perp

    return (g + perp).reshape(g_original_shape)


class MarvellPerturbation:
    def __init__(self, p_frac='pos_frac', dynamic=False, error_prob_lower_bound=None,
                 sumKL_threshold=None, init_scale=1.0, uv_choice='uv', verbose=True):
        if dynamic and (error_prob_lower_bound is not None):
            sumKL_threshold = (2 - 4 * error_prob_lower_bound) ** 2

        self.p_frac = p_frac
        self.dynamic = dynamic
        self.error_prob_lower_bound = error_prob_lower_bound
        self.sumKL_threshold = sumKL_threshold
        self.init_scale = init_scale
        self.uv_choice = uv_choice
        self.solver_log = []

        if verbose:
            print('p_frac', p_frac)
            print('dynamic', dynamic)
            if dynamic and error_prob_lower_bound is not None:
                print('error_prob_lower_bound', error_prob_lower_bound)
                print('implied sumKL_threshold', sumKL_threshold)
            elif dynamic:
                print('using sumKL_threshold', sumKL_threshold)
            print('init_scale', init_scale)
            print('uv_choice', uv_choice)

    def __call__(self, g, y):
        g_original_shape = g.shape
        g = g.reshape(g_original_shape[0], -1)
        y = y.reshape(-1)
        y_bool = y > 0.5

        pos_g = g[y_bool]
        pos_g_mean = torch.mean(pos_g, dim=0, keepdim=True)
        pos_coordinate_var = torch.mean((pos_g - pos_g_mean) ** 2, dim=0)
        neg_g = g[~y_bool]
        neg_g_mean = torch.mean(neg_g, dim=0, keepdim=True)
        neg_coordinate_var = torch.mean((neg_g - neg_g_mean) ** 2, dim=0)

        avg_pos_coordinate_var = torch.mean(pos_coordinate_var)
        avg_neg_coordinate_var = torch.mean(neg_coordinate_var)

        g_diff = pos_g_mean - neg_g_mean
        g_diff_norm = float(torch.norm(g_diff).item())

        if pos_g.shape[0] == 0 or neg_g.shape[0] == 0 or g_diff_norm == 0.0:
            return g.reshape(g_original_shape)

        if self.uv_choice == 'uv':
            u = float(avg_neg_coordinate_var)
            v = float(avg_pos_coordinate_var)
        elif self.uv_choice == 'same':
            u = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
            v = float(avg_neg_coordinate_var + avg_pos_coordinate_var) / 2.0
        elif self.uv_choice == 'zero':
            u, v = 0.0, 0.0
        else:
            raise ValueError(f"unknown uv_choice {self.uv_choice}")

        d = float(g.shape[1])

        if self.p_frac == 'pos_frac':
            p = float(y_bool.float().sum().item() / len(y))
        else:
            p = float(self.p_frac)

        scale = self.init_scale

        solver_start = time.perf_counter()
        lam10, lam20, lam11, lam21 = None, None, None, None
        while True:
            P = scale * g_diff_norm ** 2
            lam10, lam20, lam11, lam21, sumKL = \
                solve_isotropic_covariance(
                    u=u,
                    v=v,
                    d=d,
                    g=g_diff_norm ** 2,
                    p=p,
                    P=P,
                    lam10_init=lam10,
                    lam20_init=lam20,
                    lam11_init=lam11,
                    lam21_init=lam21)

            if not self.dynamic or sumKL <= self.sumKL_threshold:
                break

            scale *= 1.5

        solver_ms = (time.perf_counter() - solver_start) * 1000.0

        self.solver_log.append({
            'solver_ms': solver_ms, 'd': d,
            'u': u, 'v': v, 'g': g_diff_norm ** 2, 'p': p, 'scale': scale, 'P': P,
            'lam10': lam10, 'lam20': lam20, 'lam11': lam11, 'lam21': lam21,
            'sumKL_before': symKL_objective(lam10=0.0, lam20=0.0, lam11=0.0, lam21=0.0,
                                            u=float(avg_neg_coordinate_var),
                                            v=float(avg_pos_coordinate_var),
                                            d=d, g=g_diff_norm ** 2),
            'sumKL_after': sumKL,
            'error_prob_lower_bound': 0.5 - math.sqrt(sumKL) / 4,
        })

        perturbed_g = g.clone()
        y_float = y_bool.to(g.dtype)
        batch = y.shape[0]

        perturbed_g += (torch.randn(batch, device=g.device, dtype=g.dtype) * y_float).reshape(-1, 1) \
            * g_diff * (math.sqrt(max(lam11 - lam21, 0.0)) / g_diff_norm)

        if lam21 > 0.0:
            perturbed_g += torch.randn_like(g) * y_float.reshape(-1, 1) * math.sqrt(lam21)

        perturbed_g += (torch.randn(batch, device=g.device, dtype=g.dtype) * (1 - y_float)).reshape(-1, 1) \
            * g_diff * (math.sqrt(max(lam10 - lam20, 0.0)) / g_diff_norm)

        if lam20 > 0.0:
            perturbed_g += torch.randn_like(g) * (1 - y_float).reshape(-1, 1) * math.sqrt(lam20)

        return perturbed_g.reshape(g_original_shape)


class LabelProtectionDefense:
    def __init__(self, method='no_noise', **params):
        if method not in OFFICIAL_NAME_MAP:
            raise ValueError(f"unknown defense '{method}', choose from {sorted(OFFICIAL_NAME_MAP)}")
        self.method = OFFICIAL_NAME_MAP[method]
        self.params = params
        self._marvell = None

        if self.method == 'marvell':
            self._marvell = MarvellPerturbation(**params)
        elif self.method == 'iso':
            self.ratio = float(params.get('ratio', 1.0))
        elif self.method == 'perp':
            self.lower = float(params.get('lower', 1.0))
            self.upper = float(params.get('upper', 5.0))

    @property
    def solver_log(self):
        return self._marvell.solver_log if self._marvell is not None else []

    def tag(self):
        if self.method == 'marvell':
            return f"marvell_s{self._marvell.init_scale:g}"
        if self.method == 'iso':
            return f"iso_r{self.ratio:g}"
        if self.method == 'perp':
            return f"perp_{self.lower:g}-{self.upper:g}"
        return self.method

    @torch.no_grad()
    def perturb(self, grad, labels):
        if self.method == 'no_noise':
            return no_noise(grad)
        if self.method == 'max_norm':
            return max_norm_perturb(grad)
        if self.method == 'iso':
            return iso_gaussian_perturb(grad, ratio=self.ratio)
        if self.method == 'perp':
            return perp_perturb(grad, lower=self.lower, upper=self.upper)
        return self._marvell(grad, labels)


def _sync(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


class _NoiseLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, owner):
        ctx.owner = owner
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.owner.apply_defense(grad_output), None


class LabelProtectedServer(nn.Module):
    def __init__(self, server_model, defense, observer=None):
        super().__init__()
        self.server_model = server_model
        self.defense = defense
        self.observer = observer
        self.current_labels = None
        self.defense_ms = []

    def set_labels(self, labels):
        self.current_labels = labels.detach().float()

    def apply_defense(self, grad):
        labels = self.current_labels
        if labels is None or labels.shape[0] != grad.shape[0]:
            return grad
        _sync(grad)
        start = time.perf_counter()
        perturbed = self.defense.perturb(grad, labels)
        _sync(grad)
        self.defense_ms.append((time.perf_counter() - start) * 1000.0)
        if self.observer is not None:
            self.observer(grad.detach(), perturbed.detach(), labels)
        return perturbed

    def forward(self, smashed):
        if torch.is_grad_enabled() and smashed.requires_grad:
            smashed = _NoiseLayerFunction.apply(smashed, self)
        return self.server_model(smashed)


class LabelAwareCriterion(nn.Module):
    def __init__(self, criterion, protected_server):
        super().__init__()
        self.criterion = criterion
        self.protected_server = protected_server

    def forward(self, logits, labels):
        self.protected_server.set_labels(labels)
        return self.criterion(logits, labels)


def attach_label_protection(attack, defense, observer=None):
    protected = LabelProtectedServer(attack.server_model, defense, observer)
    attack.server_model = protected
    attack.criterion = LabelAwareCriterion(attack.criterion, protected)
    return protected