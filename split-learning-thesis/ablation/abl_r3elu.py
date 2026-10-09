import torch

from config import Config
from all_defences.r3elu_defense import R3eLUMechanism, clip_k, rr_probability, sample_laplace
import abl_common as C


class AblR3eLUMechanism(R3eLUMechanism):
    def __init__(self, num_features, use_rr=True, use_laplace=True, use_clipk=True, sensitivity='global', **kw):
        super().__init__(num_features, **kw)
        self.use_rr, self.use_laplace, self.use_clipk, self.sensitivity = use_rr, use_laplace, use_clipk, sensitivity

    def laplace_scale(self, C_):
        _, eps_l = self._budgets()
        sens = 2.0 * self.K * C_ if self.sensitivity == 'global' else 2.0 * C_
        return sens / eps_l

    def _rr_keep(self, ratio, eps_p):
        if not self.use_rr:
            return torch.ones_like(ratio, dtype=torch.bool)
        return torch.rand_like(ratio) < rr_probability(ratio, eps_p, self.K)

    @torch.no_grad()
    def forward_perturb(self, v):
        shape = v.shape
        flat = v.reshape(shape[0], -1)
        eps_p, _ = self._budgets()
        v_hat = clip_k(flat, self.K, self.C) if self.use_clipk else flat.clamp(min=0.0)
        if self.dynamic_budget and self.importance.ready():
            ratio = self.importance.ratio().unsqueeze(0).expand_as(v_hat)
        else:
            ratio = v_hat / (v_hat.abs().amax(dim=1, keepdim=True) + 1e-12)
        keep = self._rr_keep(ratio, eps_p)
        if self.use_laplace:
            scale = self.laplace_scale(self.C)
            if self.dynamic_budget and self.importance.ready():
                scale = scale / self.importance.laplace_weight().unsqueeze(0)
            noisy = torch.clamp(v_hat + sample_laplace(v_hat.shape, scale, v.device), min=0.0)
        else:
            noisy = v_hat
        return torch.where(keep, noisy, torch.zeros_like(noisy)).reshape(shape)

    @torch.no_grad()
    def backward_perturb(self, delta):
        shape = delta.shape
        flat = delta.reshape(shape[0], -1)
        eps_p, _ = self._budgets()
        abs_hat = clip_k(flat.abs(), self.K, self.C_backward) if self.use_clipk else flat.abs()
        ratio = abs_hat / (abs_hat.amax(dim=1, keepdim=True) + 1e-12)
        keep = self._rr_keep(ratio, eps_p)
        out = torch.where(keep, torch.sign(flat) * abs_hat, torch.zeros_like(abs_hat))
        if self.use_laplace:
            out = out + sample_laplace(out.shape, self.laplace_scale(self.C_backward), delta.device)
        return out.reshape(shape)


class R3eLUAblation:
    name = 'r3elu'
    attacks = ['whitebox', 'ae_decoder', 'label_leakage', 'fsha']

    def __init__(self, epsilon=1.0, sensitivity='global'):
        self.epsilon, self.sensitivity = epsilon, sensitivity
        base = dict(protect_forward=True, protect_backward=True, dynamic_budget=True,
                    use_rr=True, use_laplace=True, use_clipk=True, sensitivity=sensitivity)
        self.variants = {
            'full': dict(base),
            'off': dict(base, protect_forward=False, protect_backward=False),
            'no_rr': dict(base, use_rr=False),
            'no_laplace': dict(base, use_laplace=False),
            'no_clipk': dict(base, use_clipk=False),
            'no_forward': dict(base, protect_forward=False),
            'no_backward': dict(base, protect_backward=False),
            'no_dynamic_budget': dict(base, dynamic_budget=False),
            'per_coordinate_sensitivity': dict(base, sensitivity='per_coord'),
        }

    def mech(self, cfg, nf, device):
        return AblR3eLUMechanism(nf, epsilon=self.epsilon, K=nf // 2, C=10.0, device=device, **cfg)

    def _trained(self, variant, ctx):
        key = ('r3elu', self.epsilon, self.sensitivity, variant)
        if key in ctx['cache']:
            return ctx['cache'][key]
        from all_split_learning.r3elu_split_learning import R3eLUSplitLearningTrainer
        dev = ctx['device']
        c, s, _ = C.load_vanilla(dev)
        mech = self.mech(self.variants[variant], C.n_features(c, dev), dev)
        old = Config.EPOCHS
        Config.EPOCHS = C.SETTINGS['def_epochs']
        tr = R3eLUSplitLearningTrainer(c, s, ctx['train'], ctx['test'], mech, tag=f"abl_r3elu_eps{self.epsilon:g}_{variant}")
        tr._save_checkpoint = lambda *a, **k: None
        tr.train()
        Config.EPOCHS = old
        acc = tr._evaluate()
        ctx['cache'][key] = (c, mech, acc)
        return ctx['cache'][key]

    def run(self, variant, attack, ctx):
        dev = ctx['device']
        if attack == 'label_leakage':
            from defence_runner.run_r3elu_evaluation import R3eLUProtectedServer
            cfg = self.variants[variant]

            def wrap(server, client):
                return R3eLUProtectedServer(server, self.mech(cfg, C.n_features(client, dev), dev))
            return C.atk_label_leakage(wrap, ctx['train'].dataset, dev)
        if attack == 'fsha':
            r = self._fsha(variant, ctx)
            r['acc'] = self._trained(variant, ctx)[2]
            return r
        client, mech, acc = self._trained(variant, ctx)
        view = (lambda z, y: mech.forward_perturb(z)) if mech.protect_forward else (lambda z, y: z)
        if attack == 'whitebox':
            r = C.atk_whitebox(client, view, ctx['test'], dev)
        else:
            r = C.atk_ae(client, view, ctx['train'], ctx['test'], dev)
        r['acc'] = acc
        return r

    def _fsha(self, variant, ctx):
        from defence_runner.run_splitfss_evaluation import DefendedFSHAAttack
        dev = ctx['device']
        cfg = self.variants[variant]
        probe = C.build_client(dev)
        mech = self.mech(cfg, C.n_features(probe, dev), dev)
        del probe

        class R3eLUFSHAAttack(DefendedFSHAAttack):
            def _observed(self, smashed):
                out = smashed + (self.view_fn(smashed) - smashed).detach()
                if mech.protect_backward and out.requires_grad:
                    observed = out.detach()

                    def hook(g):
                        mech.importance.update(observed, g)
                        return mech.backward_perturb(g)
                    out.register_hook(hook)
                return out

        view = (lambda z: mech.forward_perturb(z)) if mech.protect_forward else (lambda z: z)
        attack = R3eLUFSHAAttack(client_model=C.build_client(), in_channels=C.in_ch(), dataset=Config.DATASET,
                                 pilot_builder=C.build_client, critic_iters=5, view_fn=view,
                                 tag=f"abl_r3elu_eps{self.epsilon:g}_{variant}")
        attack.hijack(C.Tapped(ctx['train']), ctx['test'], epochs=C.SETTINGS['fsha_epochs'])
        s = attack.reconstruct(ctx['train'], num_images=C.SETTINGS['max_images'])
        return {'ssim': float(s['ssim']), 'psnr': float(s['psnr'])}