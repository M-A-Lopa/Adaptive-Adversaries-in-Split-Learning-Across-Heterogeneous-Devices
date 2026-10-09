import torch

from all_defences.protoguard_sl_defense import ProtoGuardSLDefense
import abl_common as C


class AblProtoGuard(ProtoGuardSLDefense):
    def __init__(self, num_classes, alpha=0.5, stat='median', relation='all', reference='class',
                 space='consistency', action='replace'):
        super().__init__(num_classes=num_classes, alpha=alpha)
        self.stat, self.relation, self.reference, self.space, self.action = stat, relation, reference, space, action
        self.global_ref = None

    def _center(self, x):
        return x.median(dim=0).values if self.stat == 'median' else x.mean(dim=0)

    def _features(self, flat, c):
        if self.space == 'raw':
            return flat - self.prototypes[c]
        v = self._consistency_vectors(flat, self._proto_matrix)
        if self.relation == 'own':
            v = v[:, self._proto_classes.index(c)].unsqueeze(1)
        return v

    def _ref(self, c):
        if self.space == 'raw':
            return None
        return self.global_ref if self.reference == 'global' else self.reference_patterns[c]

    def _score(self, flat, c):
        f = self._features(flat, c)
        ref = self._ref(c)
        return f.norm(dim=1) if ref is None else (f - ref).norm(dim=1)

    def fit_embeddings(self, flat, labels):
        by = {c: flat[labels == c] for c in range(self.num_classes) if (labels == c).any()}
        self.prototypes = {c: self._center(e) for c, e in by.items()}
        self._proto_classes = sorted(self.prototypes)
        self._proto_matrix = torch.stack([self.prototypes[c] for c in self._proto_classes])
        self.reference_patterns = {}
        if self.space != 'raw':
            allv = []
            for c, e in by.items():
                v = self._consistency_vectors(e, self._proto_matrix)
                if self.relation == 'own':
                    v = v[:, self._proto_classes.index(c)].unsqueeze(1)
                self.reference_patterns[c] = self._center(v)
                allv.append(v)
            self.global_ref = self._center(torch.cat(allv))
        self.calibration_scores = {c: self._score(e, c).sort().values for c, e in by.items()}
        self.fitted = True
        return self

    def fit(self, client_model, loader, device, max_batches=50, poison_fn=None):
        client_model.eval()
        fl, lb = [], []
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= max_batches:
                    break
                x, y = b[0].to(device), b[1].to(device)
                if poison_fn is not None:
                    x, y, _ = poison_fn(x, y)
                fl.append(self._flatten(client_model(x)).cpu())
                lb.append(y.cpu())
        client_model.train()
        return self.fit_embeddings(torch.cat(fl), torch.cat(lb))

    def flagged(self, z, labels):
        if not self.fitted or labels is None:
            return None
        flat = self._flatten(z).detach().cpu()
        lab = labels.detach().cpu()
        mask = torch.zeros(len(lab), dtype=torch.bool)
        for c in self._proto_classes:
            m = lab == c
            if not m.any() or c not in self.calibration_scores:
                continue
            s = self._score(flat[m], c)
            calib = self.calibration_scores[c]
            ge = len(calib) - torch.searchsorted(calib, s, right=False)
            p = (ge.float() + 1.0) / (len(calib) + 1.0)
            mask[m.nonzero(as_tuple=True)[0][p <= self.alpha]] = True
        return mask

    def replace(self, z, labels):
        mask = self.flagged(z, labels)
        if mask is None or not mask.any():
            return z
        out = z.clone()
        lab = labels.detach().cpu()
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            p = self.prototypes[int(lab[i])].to(z.device, z.dtype)
            out[i] = p.view(-1, *([1] * (z.dim() - 2))).expand_as(z[i]) if z.dim() > 2 else p
        return out

    def observed(self, z, labels):
        return self.replace(z, labels) if self.action == 'replace' else z


class TestTimeFilter:
    def __init__(self, pg):
        self.pg = pg

    def protect(self, z, labels):
        return self.pg.replace(z, labels)


class ProtoGuardAblation:
    name = 'protoguard'
    attacks = ['whitebox', 'ae_decoder', 'fsha', 'poison_client']

    def __init__(self, alpha=0.5):
        base = dict(alpha=alpha, stat='median', relation='all', reference='class', space='consistency',
                    action='replace', fit='clean', filter_at_test=True)
        self.variants = {
            'full': base,
            'off': None,
            'mean_prototype': dict(base, stat='mean'),
            'no_consistency_raw_distance': dict(base, space='raw'),
            'own_class_relation_only': dict(base, relation='own'),
            'global_reference': dict(base, reference='global'),
            'action_drop_paper': dict(base, action='drop', filter_at_test=False),
            'no_test_time_filter': dict(base, filter_at_test=False),
            'fit_on_poisoned_stream': dict(base, fit='stream'),
            'alpha_0.3': dict(base, alpha=0.3),
            'alpha_0.7': dict(base, alpha=0.7),
            'alpha_0.9': dict(base, alpha=0.9),
        }

    def make(self, cfg):
        keys = ('alpha', 'stat', 'relation', 'reference', 'space', 'action')
        return AblProtoGuard(num_classes=10, **{k: cfg[k] for k in keys})

    def run(self, variant, attack, ctx):
        dev = ctx['device']
        cfg = self.variants[variant]
        if attack == 'poison_client':
            return self._backdoor(variant, cfg, ctx)
        key = ('protoguard', 'vanilla')
        if key not in ctx['cache']:
            ctx['cache'][key] = C.load_vanilla(dev)
        client, server, _ = ctx['cache'][key]
        if cfg is None:
            view = lambda z, y: z
            acc = C.accuracy(client, server, C.Tapped(ctx['test']), dev)
        else:
            pg = self.make(cfg).fit(client, ctx['train'], dev)
            view = pg.observed
            tv = (lambda z: pg.observed(z, C.HOLDER.get(z.shape[0], z.device))) if cfg['filter_at_test'] else None
            acc = C.accuracy(client, server, C.Tapped(ctx['test']), dev, view=tv)
        if attack == 'whitebox':
            r = C.atk_whitebox(client, view, ctx['test'], dev)
        elif attack == 'ae_decoder':
            r = C.atk_ae(client, view, ctx['train'], ctx['test'], dev)
        else:
            r = C.atk_fsha(view, ctx['train'], ctx['test'], tag=f"abl_protoguard_{variant}")
        r['acc'] = acc
        return r

    def _backdoor(self, variant, cfg, ctx):
        dev = ctx['device']
        if cfg is None:
            return C.atk_poison_client(None, ctx['train'], ctx['test'], dev, 'abl_pg_off')
        pg = self.make(cfg)
        holder = {}

        def wrap(server, atk):
            pf = atk._poison_batch if cfg['fit'] == 'stream' else None
            pg.fit(atk.client_model, ctx['train'], dev, poison_fn=pf)
            if cfg['action'] == 'replace':
                w = C.ReplaceServer(server, pg.replace, eval_view=None)
            else:
                w = C.DropServer(server, pg.flagged, eval_view=None)
            holder['w'] = w
            return w

        test_filter = TestTimeFilter(pg) if cfg['filter_at_test'] and cfg['action'] == 'replace' else None
        r = C.atk_poison_client(wrap, ctx['train'], ctx['test'], dev, f'abl_pg_{variant}', eval_defense=test_filter)
        w = holder.get('w')
        if isinstance(w, C.DropServer):
            r['rejected_pct'] = 100.0 * w.rejected / max(w.seen, 1)
        return r