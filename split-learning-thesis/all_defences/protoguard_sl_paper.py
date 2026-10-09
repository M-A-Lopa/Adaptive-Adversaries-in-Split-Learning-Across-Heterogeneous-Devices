import torch
import torch.nn.functional as F


class ProtoGuardSLPaper:
    def __init__(self, num_classes=10, alpha=0.5, window=5000, stat='median', relation='all',
                 reference='class', space='consistency', conformal='class'):
        assert stat in ('median', 'mean')
        assert relation in ('all', 'own')
        assert reference in ('class', 'global')
        assert space in ('consistency', 'raw')
        assert conformal in ('class', 'global')
        self.num_classes = num_classes
        self.alpha = float(alpha)
        self.window = int(window)
        self.stat, self.relation, self.reference = stat, relation, reference
        self.space, self.conformal = space, conformal
        self.buf_e = None
        self.buf_y = None
        self.seen = 0
        self.rejected = 0

    def _center(self, x):
        return x.median(dim=0).values if self.stat == 'median' else x.mean(dim=0)

    def _push(self, e, y):
        if self.buf_e is None or self.buf_e.shape[1] != e.shape[1] or self.buf_e.device != e.device:
            self.buf_e, self.buf_y = e, y
        else:
            self.buf_e = torch.cat([self.buf_e, e])[-self.window:]
            self.buf_y = torch.cat([self.buf_y, y])[-self.window:]

    def _prototypes(self, E, Y):
        classes = torch.unique(Y)
        protos = torch.stack([self._center(E[Y == c]) for c in classes])
        index = torch.full((self.num_classes,), -1, dtype=torch.long, device=E.device)
        index[classes] = torch.arange(len(classes), device=E.device)
        return protos, index

    def _features(self, E, Y, protos, index):
        if self.space == 'raw':
            return E - protos[index[Y]]
        V = F.normalize(E, dim=1) @ F.normalize(protos, dim=1).T
        if self.relation == 'own':
            V = V.gather(1, index[Y].unsqueeze(1))
        return V

    def _scores(self, Fv, Y):
        if self.space == 'raw':
            return Fv.norm(dim=1)
        if self.reference == 'global':
            return (Fv - self._center(Fv)).norm(dim=1)
        S = torch.empty(Fv.shape[0], device=Fv.device)
        for c in torch.unique(Y):
            m = Y == c
            S[m] = (Fv[m] - self._center(Fv[m])).norm(dim=1)
        return S

    def _p_values(self, S, Y, n_new):
        s_new, y_new = S[-n_new:], Y[-n_new:]
        p = torch.empty(n_new, device=S.device)
        if self.conformal == 'global':
            ref = S
            p[:] = ((ref.unsqueeze(0) >= s_new.unsqueeze(1)).sum(1).float() + 1.0) / (len(ref) + 1.0)
            return p
        for c in torch.unique(y_new):
            m = y_new == c
            ref = S[Y == c]
            p[m] = ((ref.unsqueeze(0) >= s_new[m].unsqueeze(1)).sum(1).float() + 1.0) / (len(ref) + 1.0)
        return p

    @torch.no_grad()
    def reject(self, z, labels):
        if labels is None:
            return None
        e = z.detach().reshape(z.shape[0], -1).float()
        y = labels.detach().to(e.device).long().reshape(-1)
        if y.shape[0] != e.shape[0]:
            return None
        self._push(e, y)
        E, Y = self.buf_e, self.buf_y
        protos, index = self._prototypes(E, Y)
        S = self._scores(self._features(E, Y, protos, index), Y)
        p = self._p_values(S, Y, e.shape[0])
        mask = p <= self.alpha
        self.seen += e.shape[0]
        self.rejected += int(mask.sum())
        return mask

    def summary(self):
        return {'seen': self.seen, 'rejected': self.rejected,
                'rejected_pct': 100.0 * self.rejected / max(self.seen, 1)}