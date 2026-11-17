"""
Undefended attack suite in the MULTI-CLIENT setup (SplitFed, util/multiclient.py).

Nothing of the old code is changed.  The attacks are the existing ones: this file imports
defence_runner/main_final.py (the single-client suite) and all_attacks/, and only hands them the
multi-client model and the victim client's data.

Per configuration (dataset, model, cut):
  1. victim model : SplitFed with NUM_CLIENTS clients for --sl-rounds rounds (default 50, like the
                    50 single-client epochs; one round = the whole training set once), FedAvg after
                    every round, best test accuracy kept.  Trained once, then loaded (resume works).
  2. floors       : the no-information controls of main_final.py, computed on the victim's data.
  3. attacks      : the server attacks VICTIM_CLIENT (default 0); the attacker sees only that client's
                    images, smashed data and gradients:
       whitebox, ae_decoder, fora   the trained SplitFed model + the victim's images
       fsha, label_leakage,         the server attacks the victim's own training, on the victim's
       similarity_label             images (as in main_final.py; the other clients are not simulated
                                    inside these attacks)
       backdoor_client              MALICIOUS_CLIENT poisons its batches from round POISON_START in a
                                    fresh SplitFed run of --bd-rounds rounds (default 10, like the
                                    10 single-client epochs) next to the honest clients, with the
                                    thesis trigger code (BackdoorPoisonAttack._poison_batch: 5 % of each
                                    of its batches, 4x4 white patch, target class 0); ASR / CDA with
                                    BackdoorPoisonAttack.evaluate
       backdoor_server              Tajalli et al.: the server trains its surrogate client while ALL
                                    clients train with it (BackdoorPoisonAttack mode 'server')
       backdoor_sfi                 SFI (Yu et al.): steal / finetune / implant, the implant guides
                                    ALL clients (SFIBackdoorAttack)
         For these two the attack code runs unchanged: its training loop gets a "multi-client
         pass" that hands it the clients' batches one after another (switching the attack's client
         model and optimiser to that client) and averages the client models (FedAvg) after every
         pass.  ASR / CDA are measured on the averaged client model.
       membership                   Shokri et al.: the target model AND every shadow model are
                                    SplitFed models (their training records split over the clients
                                    like the main split); the attack is scored on the members of
                                    VICTIM_CLIENT against as many non-members
       property                     Mao et al.: the trained SplitFed model; the attacker's auxiliary
                                    records are test images, the evaluation records are the
                                    victim client's own images
       pcat                         PCAT: the server steals the victim client's functionality while
                                    that client trains on its own images (as in run_new_attacks_all.py)

Results: results/mc/<setting>/undefended_suite/<dataset>/<model>/cut<k>/  (same files and report as
the single-client suite, e.g. results/mc/K5_iid/undefended_suite/undefended_attack_suite.pdf).
Checkpoints: checkpoints/mc/<setting>/undefended_suite/...      <setting> = K5_iid, K5_noniid0.5, ...

Run from the project root:
    python multiclient_attacks_run/mc_suite.py --datasets CIFAR10 --models KAGN --cuts 1 --attacks ae_decoder
    python multiclient_attacks_run/mc_suite.py --split noniid
    python multiclient_attacks_run/mc_suite.py --quick            (a few minutes, checks that everything runs)
Resume: run the same command again; finished attacks are skipped, an interrupted SplitFed training
continues from its last round.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, 'defence_runner')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import argparse                      # noqa: E402
import contextlib                    # noqa: E402
import shutil                        # noqa: E402
import time                          # noqa: E402
import traceback                     # noqa: E402
from datetime import datetime        # noqa: E402
from types import SimpleNamespace    # noqa: E402

import numpy as np                   # noqa: E402
import pandas as pd                  # noqa: E402
import torch                         # noqa: E402

from config import Config            # noqa: E402
import main_final as MF              # noqa: E402  the single-client suite (attacks, floors, report)
from all_attacks import backdoor_poison_attack as BD   # noqa: E402
from all_attacks.sfi_backdoor_attack import SFIBackdoorAttack, SFI_LR          # noqa: E402
from all_attacks.membership_inference_attack import ShokriMembershipInference  # noqa: E402
from all_attacks.property_inference_attack import MaoPropertyInference, PROPERTIES  # noqa: E402
from all_attacks.pcat_attack import PCATAttack          # noqa: E402
from util.split_env import load_raw_datasets, stack_images, get_targets as raw_targets  # noqa: E402
import copy                          # noqa: E402
from torch.utils.data import DataLoader, Subset         # noqa: E402
from util import multiclient as MC   # noqa: E402

ATTACKS = ['whitebox', 'ae_decoder', 'fsha', 'fora', 'label_leakage', 'similarity_label', 'backdoor_client',
           'backdoor_server', 'backdoor_sfi', 'membership', 'property', 'pcat']
LABEL = dict(MF.ATTACK_LABEL, membership='Membership inference (Shokri et al.)',
             property='Property inference (Mao et al.)', pcat='PCAT model stealing')
# settings of the three attacks of defence_runner/new_attacks/run_new_attacks_all.py (its defaults)
NEW_FULL = dict(num_shadows=10, target_size=None, n_aux=5000, n_eval=10000, pia_arch='mlp',
                aux_per_class=None, start_batch=100, vanilla_iters=5000, epochs=Config.EPOCHS)
NEW_QUICK = dict(num_shadows=1, target_size=1000, n_aux=500, n_eval=1000, pia_arch='mlp',
                 aux_per_class=5, start_batch=5, vanilla_iters=30, epochs=1)
HOW = {
    'whitebox': 'trained SplitFed model, test images',
    'ae_decoder': 'trained SplitFed model, decoder trained on the victim client images',
    'fora': 'trained SplitFed model, victim client images',
    'fsha': 'server hijacks the victim client training, victim client images',
    'label_leakage': 'server reads the victim client gradients, victim client images',
    'similarity_label': 'server clusters the victim client gradients, victim client images',
    'backdoor_client': 'malicious client poisons in a SplitFed run next to the honest clients',
    'backdoor_server': 'server surrogate backdoor (Tajalli) while all clients train, FedAvg every pass',
    'backdoor_sfi': 'server SFI implant while all clients train, FedAvg every pass',
    'membership': 'target and shadow models trained with SplitFed; members of the victim client scored',
    'property': 'trained SplitFed model; evaluated on the victim client images',
    'pcat': 'server steals the victim client functionality while it trains on its own images; server aux data from outside the victim client',
}


# --------------------------------------------------------------------------- #
#  Victim model: SplitFed                                                      #
# --------------------------------------------------------------------------- #
def splitfed_builders(ctx, num_classes):
    def client_builder():
        c, _ = MF.build_split(ctx, num_classes)
        return c

    def server_builder():
        _, s = MF.build_split(ctx, num_classes)
        return s
    return client_builder, server_builder


def train_or_load_splitfed(ctx, fresh):
    S, s = ctx['S'], ctx['mc']
    best = os.path.join(Config.SAVE_DIR, 'best_splitfed.pth')
    state = os.path.join(Config.SAVE_DIR, 'splitfed_state.pt')
    if fresh:
        for p in (best, state):
            if os.path.exists(p):
                os.remove(p)
    rounds = S['sl_epochs']
    sl_path = os.path.join(ctx['dir'], 'sl_training.json')
    info = MF.load_json(sl_path) or {}
    done = info.get('rounds') == rounds and os.path.exists(best) and not os.path.exists(state + '.running')
    if not done:
        cb, sb = splitfed_builders(ctx, Config.NUM_CLASSES)
        t0 = time.time()
        open(state + '.running', 'w').close()
        tr = MC.SplitFedTrainer(cb, sb(), ctx['full_train_ds'], ctx['shards'], ctx['test_loader'], rounds,
                                ctx['device'], best_path=best, state_path=state, seed=s['seed'],
                                label=f"{ctx['dataset']} {ctx['model']} cut {ctx['cut']}",
                                signature={'dataset': ctx['dataset'], 'model': ctx['model'], 'cut': ctx['cut'],
                                           'setting': MC.setting_tag(s), 'purpose': 'victim model'})
        tr.train(max_batches=ctx['max_batches'])
        os.remove(state + '.running')
        info = {'round': [h['round'] for h in tr.history], 'test_accuracy': [h['test_acc'] for h in tr.history],
                'train_loss': [h['loss'] for h in tr.history], 'train_accuracy': [h['train_acc'] for h in tr.history],
                'rounds': rounds, 'best_round': tr.best_round, 'seconds': time.time() - t0}
    client, server = MF.build_split(ctx, Config.NUM_CLASSES)
    st = torch.load(best, map_location=ctx['device'], weights_only=False)
    client.load_state_dict(st['client_state'])
    server.load_state_dict(st['server_state'])
    info['loaded_test_accuracy'] = MF.evaluate_accuracy(client, server, ctx['test_loader'], ctx['device'])
    with torch.no_grad():
        client.eval()
        z = client(torch.zeros(1, ctx['in_ch'], ctx['img'], ctx['img'], device=ctx['device']))
    info['smashed_shape'] = list(z.shape[1:])
    info['smashed_features'] = int(np.prod(z.shape[1:]))
    info['multiclient'] = dict(s, setting=MC.setting_tag(s))
    info['client_sizes'] = [int(len(p)) for p in ctx['shards']]
    MF.save_json(sl_path, info)
    print(f"\n  SplitFed victim model ready: test accuracy {info['loaded_test_accuracy']:.2f}% | "
          f"smashed shape {info['smashed_shape']}")
    return client, server, info


# --------------------------------------------------------------------------- #
#  Client backdoor in SplitFed (thesis trigger code)                           #
# --------------------------------------------------------------------------- #
class PoisonHook(MC.MCHooks):
    """Malicious client(s) poison their batches with BackdoorPoisonAttack._poison_batch; ASR every round."""

    def __init__(self, ctx, test_loader):
        S, s = ctx['S'], ctx['mc']
        self.malicious, self.start = set(s['malicious']), s['poison_start']
        self.test_loader, self.dataset = test_loader, ctx['dataset']
        trig = lambda raw: BD.apply_trigger_patch(raw, S['bd_patch'], S['bd_value'])   # noqa: E731
        # the thesis attack object's own methods, called on a light stand-in (no models, no optimisers)
        self.atk = SimpleNamespace(poison_rate=S['bd_rate'], target_label=S['bd_target'], dataset=ctx['dataset'],
                                   _trigger=trig, device=ctx['device'], client_model=None, server_model=None)
        self.atk._apply_trigger_eval = lambda imgs: BD.normalize(trig(BD.denormalize(imgs, self.dataset)),
                                                                 self.dataset)
        self.history = {'epoch': [], 'asr': [], 'cda': [], 'poisoned_images': []}
        self.n_poisoned = 0

    def batch(self, info, x, y):
        if info['client'] in self.malicious and info['round'] >= self.start:
            x, y, mask = BD.BackdoorPoisonAttack._poison_batch(self.atk, x, y)
            self.n_poisoned += int(mask.sum())
        return x, y

    def after_round(self, trainer, rnd):
        self.atk.client_model, self.atk.server_model = trainer.global_client(), trainer.server
        cda, asr = BD.BackdoorPoisonAttack.evaluate(self.atk, self.test_loader)
        self.history['epoch'].append(rnd); self.history['asr'].append(asr); self.history['cda'].append(cda)
        self.history['poisoned_images'].append(self.n_poisoned)
        print(f"    backdoor round {rnd}: ASR {asr:.2f}% | CDA {cda:.2f}% | poisoned images so far {self.n_poisoned}",
              flush=True)
        return None

    def tag(self):
        return f"poison_c{'-'.join(map(str, sorted(self.malicious)))}_r{self.start}_{self.atk.poison_rate:g}"

    def state_dict(self):
        return {'history': self.history, 'n_poisoned': self.n_poisoned}

    def load_state_dict(self, st):
        self.history, self.n_poisoned = st['history'], st['n_poisoned']


def mc_backdoor_client(ctx, _client):
    S, s = ctx['S'], ctx['mc']
    if s['num_clients'] < 2:
        raise RuntimeError("backdoor_client in the multi-client suite needs NUM_CLIENTS >= 2")
    rounds = S['bd_epochs']
    hook = PoisonHook(ctx, ctx['test_loader'])
    cb, sb = splitfed_builders(ctx, Config.NUM_CLASSES)
    state = os.path.join(Config.SAVE_DIR, 'backdoor_client_state.pt')
    tr = MC.SplitFedTrainer(cb, sb(), ctx['full_train_ds'], ctx['shards'], ctx['test_loader'], rounds,
                            ctx['device'], hooks=hook, state_path=state, seed=s['seed'],
                            label=f"client backdoor {ctx['dataset']} {ctx['model']} cut {ctx['cut']}",
                            signature={'dataset': ctx['dataset'], 'model': ctx['model'], 'cut': ctx['cut'],
                                       'setting': MC.setting_tag(s), 'purpose': 'client backdoor'})
    tr.train(max_batches=ctx['max_batches'])
    h = hook.history
    summ = {'mode': 'client', 'target_label': S['bd_target'], 'poison_rate': S['bd_rate'],
            'asr': h['asr'][-1], 'cda': h['cda'][-1], 'best_asr': max(h['asr']), 'epochs': rounds,
            'clients': s['num_clients'], 'malicious': s['malicious'], 'poison_start': s['poison_start'],
            'poisoned_images': hook.n_poisoned,
            'malicious_shard_size': int(sum(len(ctx['shards'][m]) for m in s['malicious']))}
    print(f"\n  client backdoor (multi-client): ASR {summ['asr']:.2f}% | CDA {summ['cda']:.2f}%")
    return {'summary': summ, 'history': h}, None


# --------------------------------------------------------------------------- #
#  Server backdoors (Tajalli, SFI): the attack's own loop over all clients     #
# --------------------------------------------------------------------------- #
class MultiClientPass:
    """
    Stands in for a train_loader inside an attack's own training loop.  One iteration = one SplitFed
    pass: the clients' batches interleaved; before every batch set_active(k) points the attack at
    client k's model and optimiser; after the pass the client models are averaged (FedAvg) and
    client 0 (holding the average) is made active again, so the attack evaluates the averaged model.
    """

    def __init__(self, clients, set_active, train_ds, shards, batch_size, seed, max_batches=None):
        self.clients, self.set_active = clients, set_active
        self.train_ds, self.shards = train_ds, shards
        self.batch_size, self.seed, self.max_batches = batch_size, seed, max_batches
        self.passes = 0

    def __len__(self):
        per = [len(s) // self.batch_size for s in self.shards]
        if self.max_batches is not None:
            per = [min(p, self.max_batches) for p in per]
        return max(1, sum(per))

    def __iter__(self):
        self.passes += 1
        K = len(self.shards)
        iters = {}
        for k in range(K):
            g = torch.Generator().manual_seed(int(self.seed) * 1_000_003 + 7_919 * self.passes + k)
            iters[k] = iter(DataLoader(Subset(self.train_ds, self.shards[k].tolist()), batch_size=self.batch_size,
                                       shuffle=True, drop_last=True, generator=g, num_workers=0))
        count = {k: 0 for k in range(K)}
        alive = list(range(K))
        while alive:
            for k in list(alive):
                if self.max_batches is not None and count[k] >= self.max_batches:
                    alive.remove(k)
                    continue
                try:
                    x, y = next(iters[k])
                except StopIteration:
                    alive.remove(k)
                    continue
                count[k] += 1
                self.set_active(k)
                yield x, y
        MC.fedavg(self.clients, list(range(K)))
        self.set_active(0)


def _clients_from(first, n, lr):
    clients = [first] + [copy.deepcopy(first) for _ in range(n - 1)]
    return clients, [torch.optim.Adam(c.parameters(), lr=lr) for c in clients[1:]]


def mc_backdoor_server(ctx, _client):
    S, s = ctx['S'], ctx['mc']
    c, srv = MF.build_split(ctx, Config.NUM_CLASSES)
    atk = BD.BackdoorPoisonAttack(client_model=c, server_model=srv, base_dataset=ctx['full_train_ds'],
                                  dataset=ctx['dataset'], num_classes=Config.NUM_CLASSES, mode='server',
                                  target_label=S['bd_target'], poison_rate=S['bd_rate'], patch_size=S['bd_patch'],
                                  trigger_value=S['bd_value'],
                                  surrogate_builder=lambda: MF.build_client(ctx['model'], ctx['cut'], ctx['in_ch']),
                                  model_tag=f"mc_{MC.setting_tag(s)}_{ctx['model'].lower()}")
    clients, more_opts = _clients_from(atk.client_model, s['num_clients'], 1e-3)
    opts = [atk.client_optimizer] + more_opts

    def set_active(k):
        atk.client_model, atk.client_optimizer = clients[k], opts[k]
    loader = MultiClientPass(clients, set_active, ctx['full_train_ds'], ctx['shards'], Config.BATCH_SIZE,
                             s['seed'], ctx['max_batches'])
    loader.batch_size = Config.BATCH_SIZE
    atk.pretrain_server_backdoor(loader)
    atk.train(loader, ctx['test_loader'], epochs=S['bd_epochs'])
    summ = atk.summarise()
    summ['best_asr'] = max(atk.history['asr']) if atk.history['asr'] else None
    summ['epochs'] = S['bd_epochs']
    summ['clients'] = s['num_clients']
    return {'summary': summ, 'history': atk.history}, None


def mc_backdoor_sfi(ctx, _client):
    S, s = ctx['S'], ctx['mc']
    c, srv = MF.build_split(ctx, Config.NUM_CLASSES)
    atk = SFIBackdoorAttack(client_model=c, server_model=srv, base_dataset=ctx['full_train_ds'],
                            dataset=ctx['dataset'], num_classes=Config.NUM_CLASSES, target_label=S['bd_target'],
                            surrogate_builder=lambda: MF.build_client(ctx['model'], ctx['cut'], ctx['in_ch']),
                            model_tag=f"mc_{MC.setting_tag(s)}_{ctx['model'].lower()}")
    clients, more_opts = _clients_from(atk.client, s['num_clients'], SFI_LR)
    opts = [atk.client_opt] + more_opts

    def set_active(k):
        atk.client, atk.client_opt = clients[k], opts[k]
    loader = MultiClientPass(clients, set_active, ctx['full_train_ds'], ctx['shards'], Config.BATCH_SIZE,
                             s['seed'], ctx['max_batches'])
    loader.batch_size = Config.BATCH_SIZE
    if ctx['max_batches'] is not None:                 # --quick only: fewer passes
        atk.stage1_rounds, atk.stage2_rounds = 1, 1
    atk.train(loader, ctx['test_loader'], epochs=S['sfi_epochs'])
    summ = atk.summarise()
    summ['epochs'] = S['sfi_epochs']
    summ['clients'] = s['num_clients']
    return {'summary': summ, 'history': atk.history}, None


# --------------------------------------------------------------------------- #
#  Membership, property, PCAT (defence_runner/new_attacks)                      #
# --------------------------------------------------------------------------- #
def _raw(ctx):
    if ctx.get('raw') is None:
        ctx['raw'] = load_raw_datasets(ctx['dataset'])
    return ctx['raw']


class MCMembership(ShokriMembershipInference):
    """Shokri et al. with SplitFed target and shadow models; scored on the victim client's members."""

    def __init__(self, mc, monitor_loader, max_batches, **kw):
        self.mc, self.monitor_loader, self.max_batches = mc, monitor_loader, max_batches
        super().__init__(train_fn=self._splitfed_train, **kw)

    def _assign(self, idx):
        """Split these training records over the clients (same rule as the main split)."""
        idx = np.asarray(idx)
        t = raw_targets(self.raw['train_plain'])[idx]
        alpha = None if self.mc['split'] == 'iid' else float(self.mc['alpha'])
        parts = MC.make_shards(t, self.mc['num_clients'], np.random.RandomState(self.seed), alpha,
                               min_size=2 * Config.BATCH_SIZE if alpha else 0)
        return [idx[p] for p in parts]

    def _splitfed_train(self, client, server, loader, epochs, device, desc):
        shards = self._assign(loader.dataset.indices)
        tr = MC.SplitFedTrainer(lambda: copy.deepcopy(client), server, self.raw['train_aug'], shards,
                                self.monitor_loader, epochs, device, seed=self.seed, label=desc)
        tr.train(max_batches=self.max_batches)
        client.load_state_dict(tr.global_client().state_dict())
        return tr.history

    def run(self):
        self.train_target()
        self.build_attack_dataset()
        self.train_attack_models()
        victim_in = self._assign(self.splits['target_in'])[self.mc['victim']]
        self.splits['target_in'] = victim_in
        self.splits['target_out'] = self.splits['target_out'][:len(victim_in)]
        res = self.evaluate()
        res['scored_members'] = f"client {self.mc['victim']} ({len(victim_in)} members, {len(victim_in)} non-members)"
        return res


def mc_membership(ctx, _client):
    S, s = ctx['S'], ctx['mc']
    raw = _raw(ctx)
    N = ctx['NEW']
    monitor = MF.make_loader(Subset(raw['test_plain'], list(range(min(1000, len(raw['test_plain']))))), False)
    k = N['num_shadows']
    atk = MCMembership(s, monitor, ctx['max_batches'], model_name=ctx['model'], cut_layer=ctx['cut'],
                       dataset=ctx['dataset'], raw=raw, device=ctx['device'], target_size=N['target_size'],
                       num_shadows=k, epochs=N['epochs'], seed=s['seed'], tag=f"mc_{MC.setting_tag(s)}")
    res = atk.run()
    return {'summary': res}, None


class _VictimOnly:
    """A shallow copy of a torchvision dataset holding only some records (keeps .data and .targets)."""

    @staticmethod
    def make(ds, idx):
        v = copy.copy(ds)
        v.data = ds.data[idx]
        t = ds.targets
        v.targets = t[torch.as_tensor(idx)] if torch.is_tensor(t) else np.asarray(t)[idx].tolist()
        return v


class MCProperty(MaoPropertyInference):
    """Mao et al. on the SplitFed model; the private records are the victim client's images."""

    def __init__(self, victim, victim_acc, **kw):
        self._victim, self._victim_acc = victim, victim_acc
        super().__init__(**kw)

    def get_victim(self):
        self.client, self.server = self._victim
        self.victim_acc = self._victim_acc
        print(f"  Victim: the trained SplitFed model (test accuracy {self.victim_acc:.2f}%)")


def mc_property(ctx, client):
    s, N = ctx['mc'], ctx['NEW']
    raw = _raw(ctx)
    victim_view = _VictimOnly.make(raw['train_plain'], ctx['shards'][s['victim']])
    vraw = {'train_plain': victim_view, 'test_plain': raw['test_plain'], 'train_aug': raw['train_aug']}
    acc = MF.evaluate_accuracy(client, ctx['server'], ctx['test_loader'], ctx['device'])
    rows = {}
    for prop in PROPERTIES[ctx['dataset']]:
        atk = MCProperty((client, ctx['server']), acc, model_name=ctx['model'], cut_layer=ctx['cut'],
                         dataset=ctx['dataset'], raw=vraw, device=ctx['device'], prop=prop, n_aux=N['n_aux'],
                         n_eval=N['n_eval'], arch=N['pia_arch'], seed=s['seed'], tag=f"mc_{MC.setting_tag(s)}")
        rows[prop] = atk.run()
    return {'summary': rows}, None


class MCPCAT(PCATAttack):
    """PCAT where the victim client trains on its own images; the server's public data are other records."""

    def __init__(self, victim_idx, **kw):
        self._victim_idx = np.asarray(victim_idx)
        super().__init__(**kw)

    def _make_splits(self):
        # Same as PCATAttack._make_splits, except that the server's public share (10 % of the training
        # set, from which its auxiliary records are drawn) comes from records OUTSIDE the victim
        # client, and the victim's private data is its whole shard.  (Taken from all records, the
        # public share would overlap the victim's shard, because both are drawn with the same seed.)
        targets = raw_targets(self.raw['train_plain'])
        rng = np.random.RandomState(self.seed)
        mine = np.zeros(len(targets), dtype=bool)
        mine[self._victim_idx] = True
        others = rng.permutation(np.where(~mine)[0])
        n_pub = int(round(self.public_fraction * len(targets)))
        self.public_idx, self.private_idx = others[:n_pub], np.sort(self._victim_idx)
        aux = []
        for c in range(self.num_classes):
            pool = self.public_idx[targets[self.public_idx] == c]
            aux.extend(rng.choice(pool, size=min(self.aux_per_class, len(pool)), replace=False).tolist())
        self.aux_idx = np.asarray(aux)
        self.X_aux, self.y_aux = stack_images(self.raw['train_plain'], self.aux_idx)
        self.X_aux, self.y_aux = self.X_aux.to(self.device), self.y_aux.to(self.device)
        self.aux_by_class = {c: torch.nonzero(self.y_aux == c).flatten() for c in range(self.num_classes)}


def mc_pcat(ctx, _client):
    s, N = ctx['mc'], ctx['NEW']
    raw = _raw(ctx)
    atk = MCPCAT(ctx['shards'][s['victim']], model_name=ctx['model'], cut_layer=ctx['cut'], dataset=ctx['dataset'],
                 raw=raw, device=ctx['device'], aux_per_class=N['aux_per_class'], start_batch=N['start_batch'],
                 epochs=N['epochs'], vanilla_iters=N['vanilla_iters'], seed=s['seed'],
                 tag=f"mc_{MC.setting_tag(s)}")
    res = atk.run()
    return {'summary': res, 'history': atk.history}, None


MC_RUNNERS = {'backdoor_client': mc_backdoor_client, 'backdoor_server': mc_backdoor_server,
              'backdoor_sfi': mc_backdoor_sfi, 'membership': mc_membership, 'property': mc_property,
              'pcat': mc_pcat}


# --------------------------------------------------------------------------- #
#  One configuration                                                           #
# --------------------------------------------------------------------------- #
def run_config(args, S, s, out_root, ck_root, ds, model, cut, train_full, test_ds, shards, attacks):
    d = MF.config_dir(out_root, ds, model, cut)
    os.makedirs(d, exist_ok=True)
    device = torch.device(Config.DEVICE if torch.cuda.is_available() else 'cpu')
    Config.DATASET, Config.MODEL_NAME, Config.CUT_LAYER = ds, model, cut
    Config.RESULTS_DIR = d
    Config.SAVE_DIR = os.path.join(ck_root, f"{ds}_{model}_cut{cut}")
    os.makedirs(Config.SAVE_DIR, exist_ok=True)

    tee = MF.Tee(os.path.join(d, 'run_log.txt'), sys.stdout)
    with contextlib.redirect_stdout(tee):
        print("\n" + "#" * 78)
        print(f"#  MULTI-CLIENT UNDEFENDED SUITE | {ds} | {model} | cut {cut} | {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"#  {MC.describe(s)}")
        print("#" * 78)
        MF.set_seed(s['seed'])
        victim_ds = MC.client_subset(train_full, shards, s['victim'])
        ctx = {'dataset': ds, 'model': model, 'cut': cut, 'device': device, 'S': S, 'seed': s['seed'],
               'in_ch': MF.in_channels_of(ds), 'img': MF.image_size_of(ds), 'dir': d,
               'train_ds': victim_ds, 'test_ds': test_ds,
               'train_loader': MF.make_loader(victim_ds, True), 'test_loader': MF.make_loader(test_ds, False),
               'full_train_ds': train_full, 'shards': shards, 'mc': s,
               'max_batches': args.max_batches, 'raw': None, 'NEW': NEW_QUICK if args.quick else NEW_FULL}
        fixed = []
        for x, _ in ctx['test_loader']:
            fixed.append(x)
            if sum(t.shape[0] for t in fixed) >= S['fig_images']:
                break
        ctx['fixed_x'] = torch.cat(fixed)[:S['fig_images']]
        torch.save(MF.denorm(ctx['fixed_x'], ds).cpu(), os.path.join(d, 'originals.pt'))

        try:
            client, server, _ = train_or_load_splitfed(ctx, args.fresh)
            ctx['server'] = server
        except Exception:
            err = traceback.format_exc()
            print(err)
            MF.save_json(os.path.join(d, 'sl_training.json'), {'error': err})
            tee.close()
            return False

        floors_path = os.path.join(d, 'floors.json')
        if args.fresh or not os.path.exists(floors_path) or 'error' in (MF.load_json(floors_path) or {}):
            try:
                t_fl = time.time()
                floors = MF.compute_floors(ctx, client, server)
                floors['seconds'] = time.time() - t_fl
                MF.save_json(floors_path, floors)
            except Exception:
                err = traceback.format_exc()
                print(err)
                MF.save_json(floors_path, {'error': err})

        ok = True
        for i, name in enumerate(attacks):
            res_path = os.path.join(d, f"attack_{name}.json")
            if not args.fresh and os.path.exists(res_path) and 'error' not in (MF.load_json(res_path) or {}):
                print(f"\n  [skip] {LABEL[name]} already done ({res_path})")
                continue
            print("\n" + "=" * 78)
            print(f"  ATTACK: {LABEL[name]} | {ds} | {model} | cut {cut} | {MC.setting_tag(s)}")
            print(f"  multi-client: {HOW[name]}")
            print("=" * 78)
            MF.set_seed(s['seed'] + 101 * (ATTACKS.index(name) + 1))
            t0 = time.time()
            try:
                runner = MC_RUNNERS.get(name) or MF.RUNNERS[name]
                result, recon = runner(ctx, client)
                result['seconds'] = time.time() - t0
                result['multiclient'] = dict(s, setting=MC.setting_tag(s), how=HOW[name])
                if recon is not None:
                    torch.save(recon, os.path.join(d, f"recon_{name}.pt"))
                MF.save_json(res_path, result)
                hist = result.get('history')
                if isinstance(hist, dict) and hist:
                    lens = {len(v) for v in hist.values() if isinstance(v, list)}
                    if len(lens) == 1:
                        pd.DataFrame(hist).to_csv(os.path.join(d, f"history_{name}.csv"), index=False)
                print(f"\n  [done] {LABEL[name]} in {(time.time() - t0) / 60:.1f} min")
            except Exception:
                ok = False
                err = traceback.format_exc()
                print(err)
                MF.save_json(res_path, {'error': err, 'seconds': time.time() - t0})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    tee.close()
    return ok


def main():
    ap = argparse.ArgumentParser(description="Undefended attack suite, multi-client (SplitFed) setup")
    ap.add_argument('--datasets', nargs='+', default=MF.DATASETS, choices=MF.DATASETS)
    ap.add_argument('--models', nargs='+', default=MF.MODELS, choices=MF.MODELS)
    ap.add_argument('--cuts', nargs='+', type=int, default=MF.CUTS, choices=MF.CUTS)
    ap.add_argument('--attacks', nargs='+', default=ATTACKS, choices=ATTACKS)
    ap.add_argument('--clients', type=int, default=None, help='overrides NUM_CLIENTS of config.py')
    ap.add_argument('--split', default=None, choices=list(MC.SPLITS), help='overrides DATA_SPLIT')
    ap.add_argument('--alpha', type=float, default=None, help='overrides NONIID_ALPHA')
    ap.add_argument('--victim', type=int, default=None, help='overrides VICTIM_CLIENT')
    ap.add_argument('--malicious', type=int, nargs='+', default=None, help='overrides MALICIOUS_CLIENT')
    ap.add_argument('--poison-start', type=int, default=None, help='overrides POISON_START')
    ap.add_argument('--seed', type=int, default=None, help='overrides SEED')
    ap.add_argument('--sl-rounds', type=int, default=None, help='SplitFed rounds of the victim model (default 50)')
    ap.add_argument('--bd-rounds', type=int, default=None, help='SplitFed rounds of the client backdoor (default 10)')
    ap.add_argument('--quick', action='store_true', help='tiny settings, only checks that everything runs')
    ap.add_argument('--fresh', action='store_true', help='ignore finished results and train again')
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()

    s = MC.mc_settings(num_clients=args.clients, split=args.split, alpha=args.alpha, victim=args.victim,
                       malicious=args.malicious, poison_start=args.poison_start, seed=args.seed)
    if s['num_clients'] < 2:
        sys.exit("NUM_CLIENTS is 1: that is the original single-client suite (defence_runner/main_final.py). "
                 "Use --clients 5 or set NUM_CLIENTS = 5 in config.py.")
    Config.NUM_CLIENTS = s['num_clients']
    S = dict(MF.QUICK if args.quick else MF.FULL)
    S['sl_epochs'] = args.sl_rounds or (S['sl_epochs'] if args.quick else 50)
    if args.bd_rounds:
        S['bd_epochs'] = args.bd_rounds
    S['train_subset'] = None                          # the client split always uses the full training set
    args.max_batches = 8 if args.quick else None

    tag = MC.setting_tag(s)
    name = 'undefended_suite_quick' if args.quick else 'undefended_suite'
    out_root = os.path.join(ROOT, 'results', 'mc', tag, name)
    ck_root = os.path.join(ROOT, 'checkpoints', 'mc', tag, name)
    os.makedirs(out_root, exist_ok=True)
    MF.save_json(os.path.join(out_root, 'settings.json'),
                 dict(S, multiclient=dict(s, setting=tag), datasets=args.datasets, models=args.models,
                      cuts=args.cuts, attacks=args.attacks, how=HOW))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"[mc-suite] GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"[mc-suite] {MC.describe(s)}\n[mc-suite] results -> {out_root}")

    failed = False
    if not args.report_only:
        start = time.time()
        total = len(args.datasets) * len(args.models) * len(args.cuts)
        k = 0
        for ds in args.datasets:
            train_full, test_ds = MF.load_datasets(ds)
            if args.quick:
                test_ds = torch.utils.data.Subset(test_ds, list(range(min(S['test_subset'] or 256, len(test_ds)))))
            shards = MC.get_split(ds, train_full, s)
            for m in args.models:
                for c in args.cuts:
                    k += 1
                    print(f"\n[mc-suite] configuration {k}/{total}: {ds} {m} cut {c} "
                          f"(elapsed {(time.time() - start) / 60:.1f} min)")
                    failed |= not run_config(args, S, s, out_root, ck_root, ds, m, c, train_full, test_ds,
                                             shards, args.attacks)
                    MF.build_report(out_root, S, args.datasets, args.models, args.cuts, args.quick)
        print(f"\n[mc-suite] all configurations finished in {(time.time() - start) / 60:.1f} min")
    MF.build_report(out_root, S, args.datasets, args.models, args.cuts, args.quick)
    print(f"[mc-suite] report -> {os.path.join(out_root, 'undefended_attack_suite' + ('_quick' if args.quick else '') + '.pdf')}")
    if failed:
        print("[mc-suite] some attacks failed (see the errors above and attack_*.json); the next run retries them")
        sys.exit(1)


if __name__ == '__main__':
    main()