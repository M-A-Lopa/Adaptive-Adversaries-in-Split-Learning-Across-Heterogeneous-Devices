import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import abl_common as C
from config import Config
from all_defences.triguard_defense import TriGuardSL, TriGuardTrainer


class TriGuardEvaluation:
    name = 'triguard'
    attacks = ['whitebox', 'ae_decoder', 'fsha', 'label_leakage', 'poison_client']

    def __init__(self, rounds=5, num_clients=3, epsilon=1.0, sensitivity='global', clip_c=10.0):
        self.rounds, self.num_clients = rounds, num_clients
        self.clip_c = clip_c
        self.epsilon, self.sensitivity = epsilon, sensitivity
        base = dict(share_mode='full_ring', backward_noise=True, private_heads=True)
        self.variants = {
            'full': base,
            'off': dict(share_mode='plain', backward_noise=False, private_heads=False),
            'no_secret_sharing': dict(base, share_mode='plain'),
            'bounded_mask_instead_of_ring': dict(base, share_mode='bounded'),
            'no_backward_laplace': dict(base, backward_noise=False),
            'shared_classifier_heads': dict(base, private_heads=False),
        }

    def _defense(self, cfg, num_features):
        return TriGuardSL(num_features, num_clients=self.num_clients, epsilon=self.epsilon, C=self.clip_c,
                          sensitivity=self.sensitivity, **cfg)

    def _shards(self, ctx):
        ds = ctx['train'].dataset
        idx = np.random.RandomState(ctx['seed']).permutation(len(ds))
        parts = np.array_split(idx, self.num_clients)
        loaders = [DataLoader(Subset(ds, p.tolist()), batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=False)
                   for p in parts]
        mb = C.SETTINGS['max_batches']
        return [C.Limited(l, mb) for l in loaders] if mb else loaders

    def _trainer(self, cfg, ctx):
        dev = ctx['device']
        client, server, _ = C.load_vanilla(dev)
        defense = self._defense(cfg, C.n_features(client, dev))
        return defense, TriGuardTrainer(defense, client, server, self._shards(ctx), dev, Config.LEARNING_RATE)

    def _clean(self, variant, ctx):
        key = ('triguard_clean', variant)
        if key not in ctx['cache']:
            C.seed_all(ctx['seed'])
            defense, tr = self._trainer(self.variants[variant], ctx)
            for r in range(self.rounds):
                tr.train_round()
                print(f"  [TriGuard-SL {variant}] clean round {r + 1}/{self.rounds}")
            accs = [tr.evaluate(k, ctx['test'])[0] for k in range(tr.K)]
            ctx['cache'][key] = (defense, tr.bottoms[0], float(np.mean(accs)))
        return ctx['cache'][key]

    def _backdoor(self, variant, ctx):
        from all_attacks.backdoor_poison_attack import BackdoorPoisonAttack
        C.seed_all(ctx['seed'])
        dev = ctx['device']
        defense, tr = self._trainer(self.variants[variant], ctx)
        helper = BackdoorPoisonAttack(client_model=C.build_client(dev), server_model=C.build_server(10, dev),
                                      base_dataset=ctx['train'].dataset, dataset=Config.DATASET, num_classes=10,
                                      mode='client', target_label=C.SETTINGS['poison_target'],
                                      poison_rate=C.SETTINGS['poison_rate'], patch_size=C.SETTINGS['poison_patch'],
                                      trigger_value=C.SETTINGS['poison_trigger'], model_tag='triguard_helper')
        target = C.SETTINGS['poison_target']
        for r in range(self.rounds):
            tr.train_round(poison_fn=helper._poison_batch, malicious=0)
            print(f"  [TriGuard-SL {variant}] backdoor round {r + 1}/{self.rounds}")
        results = [tr.evaluate(k, ctx['test'], trigger_fn=helper._apply_trigger_eval, target=target)
                   for k in range(tr.K)]
        honest = results[1:]
        return {'asr': float(np.mean([a for _, a in honest])), 'acc': float(np.mean([c for c, _ in honest])),
                'asr_malicious_own': float(results[0][1]), 'num_clients': tr.K}

    def run(self, variant, attack, ctx):
        dev = ctx['device']
        cfg = self.variants[variant]
        if attack == 'poison_client':
            return self._backdoor(variant, ctx)
        if attack == 'label_leakage':
            def wrap(server, client):
                return self._defense(cfg, C.n_features(client, dev)).wrap_server(server)
            return C.atk_label_leakage(wrap, ctx['train'].dataset, dev)
        defense, bottom, acc = self._clean(variant, ctx)
        view = lambda z, y: defense.server_view(z)
        if attack == 'whitebox':
            r = C.atk_whitebox(bottom, view, ctx['test'], dev)
        elif attack == 'ae_decoder':
            r = C.atk_ae(bottom, view, ctx['train'], ctx['test'], dev)
        else:
            r = C.atk_fsha(view, ctx['train'], ctx['test'], tag=f"triguard_{variant}")
        r['acc'] = acc
        return r