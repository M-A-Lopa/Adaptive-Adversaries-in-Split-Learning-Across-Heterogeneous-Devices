from all_defences.protoguard_sl_paper import ProtoGuardSLPaper
import abl_common as C
from config import Config


class ProtoGuardPaperAblation:
    name = 'protoguard_paper'
    attacks = ['poison_client']

    def __init__(self, alpha=0.5, window=5000):
        base = dict(alpha=alpha, window=window, stat='median', relation='all', reference='class',
                    space='consistency', conformal='class')
        self.variants = {
            'full': base,
            'off': None,
            'no_consistency_representation': dict(base, space='raw'),
            'no_all_class_relation': dict(base, relation='own'),
            'no_class_conditional_reference': dict(base, reference='global'),
            'no_class_conditional_conformal': dict(base, conformal='global'),
            'mean_prototype': dict(base, stat='mean'),
            'window_one_batch': dict(base, window=Config.BATCH_SIZE),
            'alpha_0.3': dict(base, alpha=0.3),
            'alpha_0.7': dict(base, alpha=0.7),
            'alpha_0.9': dict(base, alpha=0.9),
        }

    def run(self, variant, attack, ctx):
        dev = ctx['device']
        cfg = self.variants[variant]
        if cfg is None:
            return C.atk_poison_client(None, ctx['train'], ctx['test'], dev, 'abl_pgp_off')
        pg = ProtoGuardSLPaper(num_classes=10, **cfg)
        holder = {}

        def wrap(server, atk):
            holder['w'] = C.DropServer(server, pg.reject)
            return holder['w']

        r = C.atk_poison_client(wrap, ctx['train'], ctx['test'], dev, f'abl_pgp_{variant}')
        r.update(holder['w'].stats())
        return r