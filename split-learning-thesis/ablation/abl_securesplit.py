import math
import warnings

import numpy as np
import torch

import abl_common as C

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class SecureSplitPaper:
    def __init__(self, reducer='umap', d_u=2, expand='pkt', degree=2, center='median', adaptive=True,
                 alpha=1.5, seed=0):
        if reducer == 'umap' and not UMAP_AVAILABLE:
            warnings.warn("umap-learn not installed; SecureSplit reducer falls back to PCA (pip install umap-learn)")
            reducer = 'pca'
        self.reducer, self.d_u, self.expand, self.degree = reducer, d_u, expand, degree
        self.center, self.adaptive, self.alpha, self.seed = center, adaptive, alpha, seed
        self._W = None
        self.seen = 0
        self.rejected = 0

    def _reduce(self, E):
        if self.reducer == 'none':
            return E
        if self.reducer == 'pca':
            Ec = E - E.mean(0, keepdims=True)
            _, _, vt = np.linalg.svd(Ec, full_matrices=False)
            return Ec @ vt[:self.d_u].T
        nn_ = max(2, min(15, E.shape[0] - 1))
        return umap.UMAP(n_components=self.d_u, n_neighbors=nn_, random_state=self.seed,
                         init='random').fit_transform(E)

    def _expand(self, U, d):
        if self.expand == 'none':
            return U
        U = (U - U.mean(0, keepdims=True)) / (U.std(0, keepdims=True) + 1e-8)
        if self._W is None or self._W.shape != (U.shape[1], d):
            rng = np.random.RandomState(self.seed)
            self._W = rng.randn(U.shape[1], d) / math.sqrt(U.shape[1])
        return (U @ self._W + 1.0) ** self.degree

    def reject(self, z, labels=None):
        E = z.detach().reshape(z.shape[0], -1).float().cpu().numpy()
        m = E.shape[0]
        if m < 4:
            return torch.zeros(m, dtype=torch.bool)
        H = self._expand(self._reduce(E), E.shape[1])
        lam = np.median(H, axis=0) if self.center == 'median' else H.mean(0)
        dist = np.linalg.norm(H - lam, axis=1)
        R = np.sort(dist)[int(math.ceil(m / 2)) - 1]
        if self.adaptive:
            sigma = float(H.var(axis=0).mean())
            R = (1.0 + 1.0 / (self.alpha + sigma)) * R
        mask = torch.from_numpy(dist > R)
        self.seen += m
        self.rejected += int(mask.sum())
        return mask


class SecureSplitAblation:
    name = 'securesplit'
    attacks = ['poison_client']

    def __init__(self, reducer='umap'):
        base = dict(reducer=reducer, expand='pkt', center='median', adaptive=True, alpha=1.5)
        self.variants = {
            'full': base,
            'off': None,
            'variant_I_R_on_E': dict(base, reducer='none', expand='none', adaptive=False),
            'variant_II_Radp_on_E': dict(base, reducer='none', expand='none', adaptive=True),
            'variant_III_R_on_U': dict(base, expand='none', adaptive=False),
            'variant_IV_Radp_on_U': dict(base, expand='none', adaptive=True),
            'variant_V_R_on_H': dict(base, adaptive=False),
            'pca_instead_of_umap': dict(base, reducer='pca'),
            'mean_center': dict(base, center='mean'),
            'alpha_0.5': dict(base, alpha=0.5),
            'alpha_5.0': dict(base, alpha=5.0),
        }

    def run(self, variant, attack, ctx):
        cfg = self.variants[variant]
        dev = ctx['device']
        if cfg is None:
            return C.atk_poison_client(None, ctx['train'], ctx['test'], dev, 'abl_ss_off')
        ss = SecureSplitPaper(**cfg)
        r = C.atk_poison_client(lambda server, atk: C.DropServer(server, ss.reject),
                                ctx['train'], ctx['test'], dev, f'abl_ss_{variant}')
        r['rejected_pct'] = 100.0 * ss.rejected / max(ss.seen, 1)
        r['reducer_used'] = ss.reducer
        return r