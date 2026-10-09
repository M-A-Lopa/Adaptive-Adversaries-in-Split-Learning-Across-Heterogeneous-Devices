import torch

from all_defences.splitfss_defense import FixedPoint, CryptoProvider, rand_ring
import abl_common as C


class FSSViews:
    def __init__(self, precision_fractional=5, bounded_mask_scale=1.0):
        self.fp = FixedPoint(10, precision_fractional)
        self.dealer = CryptoProvider()
        self.bounded_mask_scale = bounded_mask_scale

    def _shares(self, z):
        flat = z.reshape(z.shape[0], -1)
        return self.dealer.share(self.fp.encode(flat))

    def _out(self, v, z):
        return self.fp.decode(v).float().reshape(z.shape).to(z.device)

    def share_p0(self, z, y=None):
        return self._out(self._shares(z)[0], z)

    def share_p1(self, z, y=None):
        return self._out(self._shares(z)[1], z)

    def colluding(self, z, y=None):
        s0, s1 = self._shares(z)
        return self._out(s0 + s1, z)

    def fixed_point_only(self, z, y=None):
        return self._out(self.fp.encode(z.reshape(z.shape[0], -1)), z)

    def mask_ring(self, z, y=None):
        e = self.fp.encode(z.reshape(z.shape[0], -1))
        return self._out(e + rand_ring(e.shape), z)

    def mask_bounded(self, z, y=None):
        return z + (torch.rand_like(z) * 2 - 1) * self.bounded_mask_scale * z.detach().std()

    @staticmethod
    def plain(z, y=None):
        return z


class SplitFSSAblation:
    name = 'splitfss'
    attacks = ['whitebox', 'ae_decoder', 'fsha']
    crypto_variants = {'full', 'share_p1', 'colluding', 'fixed_point_only', 'mask_ring'}

    def __init__(self):
        v = FSSViews()
        self.views = {
            'full': v.share_p0,
            'off': v.plain,
            'share_p1': v.share_p1,
            'colluding': v.colluding,
            'fixed_point_only': v.fixed_point_only,
            'mask_ring': v.mask_ring,
            'mask_bounded': v.mask_bounded,
        }
        self.variants = {k: {} for k in self.views}

    def run(self, variant, attack, ctx):
        dev = ctx['device']
        key = ('splitfss', 'client')
        if key not in ctx['cache']:
            ctx['cache'][key] = C.load_vanilla(dev)
        client, server, base_acc = ctx['cache'][key]
        view = self.views[variant]
        if attack == 'whitebox':
            r = C.atk_whitebox(client, view, ctx['test'], dev)
        elif attack == 'ae_decoder':
            r = C.atk_ae(client, view, ctx['train'], ctx['test'], dev)
        else:
            r = C.atk_fsha(view, ctx['train'], ctx['test'], tag=f"abl_splitfss_{variant}")
        if variant in self.crypto_variants or variant == 'off':
            r['acc'] = base_acc
            r['acc_note'] = 'functionality-preserving protocol: plaintext model accuracy'
        else:
            r['acc'] = C.accuracy(client, server, ctx['test'], dev, view=lambda z: view(z))
        return r