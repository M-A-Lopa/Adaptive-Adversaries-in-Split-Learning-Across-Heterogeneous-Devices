import os
import time

import numpy as np
import torch

from config import Config
from all_defences.splitml_defense import SplitMLDefense
from all_split_learning.splitml_federated_learning import (SplitMLTrainer, SplitMLClientModel,
                                                           build_splitml_clients)
import abl_common as C


def homogeneous_clients(k, device):
    out = []
    for _ in range(k):
        m = SplitMLClientModel(C.in_ch(), 10, variant=0)
        with torch.no_grad():
            m(torch.zeros(2, C.in_ch(), C.img_size(), C.img_size()))
        out.append(m.to(device))
    return out


class AblSplitMLTrainer(SplitMLTrainer):
    def __init__(self, *a, share_bottom=False, **kw):
        super().__init__(*a, **kw)
        self.share_bottom = share_bottom

    def train(self):
        for r in range(self.rounds):
            for k in range(self.K):
                self._local_epoch(k, r)
            if self.collaborative:
                t0 = time.time()
                self.federation.aggregate([c.top for c in self.clients])
                if self.share_bottom:
                    self.federation.aggregate([c.bottom for c in self.clients])
                self.fed_time += time.time() - t0
            accs = [self.evaluate_client(k) for k in range(self.K)]
            print(f"  [SplitML ablation] round {r + 1}/{self.rounds} | " +
                  " | ".join(f"C{k + 1} {a:.2f}%" for k, a in enumerate(accs)))
        return [self.evaluate_client(k) for k in range(self.K)]


class SplitMLAblation:
    name = 'splitml'
    attacks = ['whitebox', 'ae_decoder', 'poison_client']

    def __init__(self, rounds=10, num_clients=3):
        self.rounds, self.K = rounds, num_clients
        self.recon_views = {
            'full': SplitMLDefense(collusion=False).protect,
            'collusion_sk_leaked': SplitMLDefense(collusion=True).protect,
            'off': lambda z: z,
        }
        base = dict(k=num_clients, homogeneous=False, share_bottom=False, collaborative=True, dp_sigma=0.0)
        self.backdoor_cfg = {
            'full': base,
            'homogeneous_private_bottoms': dict(base, homogeneous=True),
            'share_bottom_fedavg': dict(base, homogeneous=True, share_bottom=True),
            'no_federation': dict(base, collaborative=False),
            'k2_less_dilution': dict(base, k=2),
            'k5_more_dilution': dict(base, k=5),
            'extra_dp_0.01': dict(base, dp_sigma=0.01),
        }
        self.variants = {k: {} for k in list(self.recon_views) + list(self.backdoor_cfg)}

    def variants_for(self, attack):
        if attack == 'poison_client':
            return ['full', 'off'] + [v for v in self.backdoor_cfg if v != 'full']
        return list(self.recon_views)

    def _top(self, ctx):
        key = ('splitml', 'top')
        if key in ctx['cache']:
            return ctx['cache'][key]
        dev = ctx['device']
        path = f"{Config.SAVE_DIR}/splitml_{Config.MODEL_NAME.lower()}_{Config.DATASET}.pth"
        clients = build_splitml_clients(self.K, C.in_ch(), 10, dev)
        if os.path.exists(path):
            ck = torch.load(path, map_location=dev)
            for c, st in zip(clients, ck['clients']):
                c.load_state_dict(st)
        else:
            C.seed_all(0)
            tr = SplitMLTrainer(clients, ctx['train'].dataset, ctx['test'], self.rounds, collaborative=True)
            tr.train()
        accs = []
        for c in clients:
            c.eval()
        with torch.no_grad():
            for c in clients:
                hit = tot = 0
                for x, y in ctx['test']:
                    hit += c(x.to(dev)).argmax(1).eq(y.to(dev)).sum().item()
                    tot += y.size(0)
                accs.append(100.0 * hit / max(tot, 1))
        ctx['cache'][key] = (clients[0].top, float(np.mean(accs)))
        return ctx['cache'][key]

    def _backdoor(self, variant, ctx):
        from defence_runner.run_splitml_evaluation import PoisonedLoader, backdoor_asr
        if variant == 'off':
            return C.atk_poison_client(None, ctx['train'], ctx['test'], ctx['device'], 'abl_splitml_off')
        cfg = self.backdoor_cfg[variant]
        dev = ctx['device']
        C.seed_all(0)
        clients = homogeneous_clients(cfg['k'], dev) if cfg['homogeneous'] else \
            build_splitml_clients(cfg['k'], C.in_ch(), 10, dev)
        tr = AblSplitMLTrainer(clients, ctx['train'].dataset, ctx['test'], self.rounds,
                               collaborative=cfg['collaborative'], extra_dp_sigma=cfg['dp_sigma'],
                               tag=f"abl_splitml_{variant}", share_bottom=cfg['share_bottom'])
        tr.loaders[0] = PoisonedLoader(tr.loaders[0])
        accs = tr.train()
        asr = [backdoor_asr(c, ctx['test'], dev) for c in tr.clients]
        return {'asr': float(np.mean(asr[1:])), 'asr_malicious_own': float(asr[0]),
                'acc': float(np.mean(accs[1:])), 'num_clients': cfg['k']}

    def run(self, variant, attack, ctx):
        if attack == 'poison_client':
            return self._backdoor(variant, ctx)
        top, acc = self._top(ctx)
        view = self.recon_views[variant]
        fn = lambda z, y: view(z)
        if attack == 'whitebox':
            r = C.atk_whitebox(top, fn, ctx['test'], ctx['device'])
        else:
            r = C.atk_ae(top, fn, ctx['train'], ctx['test'], ctx['device'])
        r['acc'] = acc
        return r
