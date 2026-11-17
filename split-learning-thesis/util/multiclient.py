"""
Multi-client split learning (SplitFed) shared by every attack and defence.

Nothing in the old single-client code changes.  Scripts that want the multi-client setup import
this module; with NUM_CLIENTS = 1 they keep using the old code path.

Settings (config.py, every one can also be overridden on the command line of the runners):
    NUM_CLIENTS       number of clients (1 = the old single-client setup)
    DATA_SPLIT        'iid' or 'noniid' (label skew, Dirichlet(NONIID_ALPHA) per class)
    NONIID_ALPHA      Dirichlet concentration, smaller = more skewed (default 0.5)
    VICTIM_CLIENT     client the server attacks in the privacy attacks
    MALICIOUS_CLIENT  client (or list of clients) that plants the client backdoor
    POISON_START      first round in which the malicious client poisons
    SEED              seed of the split and of the training order

What this module gives:
    mc_settings()         the settings above, filled with defaults and checked
    setting_tag()         'K1', 'K5_iid', 'K5_noniid0.5' ... used in every result folder name
    get_split()           the client split, made ONCE per (dataset, clients, split, seed) and saved to
                          results/splits/<dataset>_<tag>_seed<seed>.json, together with how many
                          images of every class each client holds.  Every attack and defence reads
                          the same file, so client 0 always holds exactly the same images
    MCHooks               the four places where attacks and defences plug in (see the class)
    SplitFedTrainer       K clients, one server, batches of the clients interleaved, FedAvg of the
                          client models after every round, per-round checkpoint (resume), the best
                          model saved in the same format as the single-client trainer, optional
                          rollback + exclusion requested by a defence

Training follows all_split_learning/split_learning.py: Adam(lr=Config.LEARNING_RATE) for the client
and the server model, ReduceLROnPlateau(patience=5, factor=0.5) stepped with the test accuracy
(called exactly as in the single-client trainer), the best test accuracy kept.
One round = every client passes once over its own shard, so one round sees the whole training
set once, like one single-client epoch.
"""
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from config import Config


SPLITS = ('iid', 'noniid')


# --------------------------------------------------------------------------- #
#  Settings                                                                    #
# --------------------------------------------------------------------------- #
def mc_settings(**overrides):
    """
    Multi-client settings from config.py with defaults, then the given overrides (None = keep).
    Checks that the roles fit the number of clients and raises a clear error otherwise.
    """
    s = {
        'num_clients': getattr(Config, 'NUM_CLIENTS', 1),
        'split': getattr(Config, 'DATA_SPLIT', 'iid'),
        'alpha': getattr(Config, 'NONIID_ALPHA', 0.5),
        'victim': getattr(Config, 'VICTIM_CLIENT', 0),
        'malicious': getattr(Config, 'MALICIOUS_CLIENT', 2),
        'poison_start': getattr(Config, 'POISON_START', 3),
        'seed': getattr(Config, 'SEED', 0),
    }
    for k, v in overrides.items():
        if k not in s:
            raise KeyError(f"unknown multi-client setting '{k}'; use one of {sorted(s)}")
        if v is not None:
            s[k] = v
    K = int(s['num_clients'])
    if K < 1:
        raise ValueError(f"NUM_CLIENTS must be at least 1, got {K}")
    s['num_clients'] = K
    s['split'] = str(s['split']).lower()
    if s['split'] not in SPLITS:
        raise ValueError(f"DATA_SPLIT must be one of {SPLITS}, got '{s['split']}'")
    if s['split'] == 'noniid' and not float(s['alpha']) > 0:
        raise ValueError(f"NONIID_ALPHA must be > 0, got {s['alpha']}")
    mal = s['malicious']
    mal = [int(m) for m in mal] if isinstance(mal, (list, tuple)) else [int(mal)]
    s['malicious'] = mal
    s['victim'] = int(s['victim'])
    if K > 1:
        bad = [c for c in [s['victim']] + mal if not 0 <= c < K]
        if bad:
            raise ValueError(f"VICTIM_CLIENT / MALICIOUS_CLIENT must be between 0 and {K - 1} for "
                             f"{K} clients; got {bad}")
        if len(mal) >= K:
            raise ValueError(f"{len(mal)} malicious clients but only {K} clients: at least one must be honest")
    return s


def setting_tag(s):
    """Folder/file tag of a setting: 'K1' (old single-client), 'K5_iid', 'K5_noniid0.5'."""
    if s['num_clients'] == 1:
        return 'K1'
    split = 'iid' if s['split'] == 'iid' else f"noniid{float(s['alpha']):g}"
    return f"K{s['num_clients']}_{split}"


def describe(s):
    if s['num_clients'] == 1:
        return "1 client (single-client split learning, the original setup)"
    split = 'IID' if s['split'] == 'iid' else f"non-IID Dirichlet({float(s['alpha']):g})"
    return (f"{s['num_clients']} clients, {split}, victim client {s['victim']}, malicious client(s) "
            f"{', '.join(map(str, s['malicious']))} from round {s['poison_start']}, seed {s['seed']}")


def min_client_size(batch_size=None):
    """Smallest shard accepted for a non-IID split (a client must fill a few batches)."""
    return 4 * (batch_size or Config.BATCH_SIZE)


def need_clients(n, what):
    """For defences that compare clients: return None if possible, else the reason it is not."""
    K = getattr(Config, 'NUM_CLIENTS', 1)
    if K < n:
        return f"{what} compares clients and needs at least {n} clients (NUM_CLIENTS = {K})"
    return None


# --------------------------------------------------------------------------- #
#  Client split                                                                #
# --------------------------------------------------------------------------- #
def targets_of(ds):
    """Labels of a torchvision dataset (or a Subset of one) as a numpy array."""
    if isinstance(ds, Subset):
        return targets_of(ds.dataset)[np.asarray(ds.indices)]
    t = getattr(ds, 'targets', None)
    if t is None:
        t = [int(ds[i][1]) for i in range(len(ds))]
    if torch.is_tensor(t):
        t = t.cpu().numpy()
    return np.asarray(t)


def make_shards(targets, K, rng, alpha=None, min_size=0):
    """
    Split the indices 0..len(targets)-1 among K clients (same rule and draws as Experiment 3).
      alpha None : IID, a random permutation cut into K equal parts
      alpha > 0  : label skew, for every class a Dirichlet(alpha) share per client; drawn again
                   (up to 200 times) until every client holds at least min_size images
    """
    n = len(targets)
    if alpha is None:
        return [np.sort(p) for p in np.array_split(rng.permutation(n), K)]
    for _ in range(200):
        parts = [[] for _ in range(K)]
        for c in np.unique(targets):
            idx = rng.permutation(np.where(targets == c)[0])
            cuts = (np.cumsum(rng.dirichlet([alpha] * K))[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"could not draw a Dirichlet({alpha}) split of {n} images over {K} clients with at "
                       f"least {min_size} images per client; use fewer clients or a larger NONIID_ALPHA")


def split_path(dataset, s, root=None):
    root = root or os.path.join(getattr(Config, 'RESULTS_ROOT', None) or './results', 'splits')
    return os.path.join(root, f"{dataset}_{setting_tag(s)}_seed{s['seed']}.json")


def get_split(dataset, train_ds, s=None, root=None, verbose=True):
    """
    The client split of this setting: a list of K index arrays into train_ds.
    Made once and saved; afterwards always read from the file, so every script uses the same split.
    """
    s = s or mc_settings()
    targets = targets_of(train_ds)
    K = s['num_clients']
    if K == 1:
        return [np.arange(len(targets))]
    path = split_path(dataset, s, root)
    if os.path.exists(path):
        with open(path) as f:
            info = json.load(f)
        if info['n_train'] != len(targets) or info['num_clients'] != K:
            raise RuntimeError(f"{path} was made for {info['n_train']} images / {info['num_clients']} clients, "
                               f"but the training set has {len(targets)} images and NUM_CLIENTS = {K}. "
                               f"Delete the file only if you really want a new split.")
        shards = [np.asarray(info['clients'][str(k)], dtype=np.int64) for k in range(K)]
        if verbose:
            print(f"  [split] loaded {path}")
    else:
        rng = np.random.RandomState(s['seed'])
        alpha = None if s['split'] == 'iid' else float(s['alpha'])
        shards = make_shards(targets, K, rng, alpha, min_size=min_client_size() if alpha else 0)
        classes = np.unique(targets)
        info = {
            'dataset': dataset, 'setting': setting_tag(s), 'seed': s['seed'], 'num_clients': K,
            'split': s['split'], 'alpha': None if s['split'] == 'iid' else float(s['alpha']),
            'n_train': int(len(targets)),
            'sizes': {str(k): int(len(p)) for k, p in enumerate(shards)},
            'class_counts': {str(k): {str(int(c)): int((targets[p] == c).sum()) for c in classes}
                             for k, p in enumerate(shards)},
            'made': time.strftime('%Y-%m-%d %H:%M:%S'),
            'clients': {str(k): p.tolist() for k, p in enumerate(shards)},
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(info, f)
        os.replace(tmp, path)
        if verbose:
            print(f"  [split] made and saved {path}")
    if verbose:
        print_split(shards, targets, s)
    small = [k for k, p in enumerate(shards) if len(p) < min_client_size()]
    if small:
        print(f"  [warn] client(s) {small} hold fewer than {min_client_size()} images")
    return shards


def print_split(shards, targets, s):
    classes = np.unique(targets)
    head = 'client  size  ' + ' '.join(f"{int(c):>5d}" for c in classes) + '   role'
    print(f"  {describe(s)}\n  images per class:\n  {head}")
    for k, p in enumerate(shards):
        role = []
        if k == s['victim']:
            role.append('victim')
        if k in s['malicious']:
            role.append('malicious')
        counts = ' '.join(f"{int((targets[p] == c).sum()):>5d}" for c in classes)
        print(f"  {k:>6d} {len(p):>5d}  {counts}   {', '.join(role)}")


def client_subset(train_ds, shards, k):
    """The training images of client k (e.g. the victim's private data for an attack)."""
    return Subset(train_ds, shards[k].tolist())


# --------------------------------------------------------------------------- #
#  Hooks                                                                       #
# --------------------------------------------------------------------------- #
class MCHooks:
    """
    Where attacks and defences plug into SplitFedTrainer.  Subclass and override what you need;
    the defaults change nothing.  `info` is a dict with 'client', 'round', 'batch' (index of the
    batch of this client in this round) and 'labels'.

      batch(info, x, y)          -> x, y   before the client model: a malicious client poisons here
      smashed(info, z)           -> z      what the server receives (DP noise, secret sharing, ...);
                                           must keep the shape of z
      grad(info, g)              -> g      cut-layer gradient sent back to the client
      after_round(trainer, rnd)  -> None or {'rollback_to': r, 'exclude': [clients], 'reason': str}
                                           called after FedAvg; a detector can roll back to the end
                                           of an earlier round (one of the kept snapshots) and
                                           exclude clients for the rest of the training
      tag()                      -> str    short name, part of the checkpoint signature
    """

    def batch(self, info, x, y):
        return x, y

    def smashed(self, info, z):
        return z

    def grad(self, info, g):
        return g

    def after_round(self, trainer, rnd):
        return None

    def tag(self):
        return 'none'


class HookChain(MCHooks):
    """Several hooks in a row (e.g. a backdoor attacker and a defence)."""

    def __init__(self, *hooks):
        self.hooks = [h for h in hooks if h is not None]

    def batch(self, info, x, y):
        for h in self.hooks:
            x, y = h.batch(info, x, y)
        return x, y

    def smashed(self, info, z):
        for h in self.hooks:
            z = h.smashed(info, z)
        return z

    def grad(self, info, g):
        for h in reversed(self.hooks):
            g = h.grad(info, g)
        return g

    def after_round(self, trainer, rnd):
        for h in self.hooks:
            out = h.after_round(trainer, rnd)
            if out:
                return out
        return None

    def tag(self):
        return '+'.join(h.tag() for h in self.hooks) or 'none'


# --------------------------------------------------------------------------- #
#  Trainer                                                                     #
# --------------------------------------------------------------------------- #
def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_state():
    st = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    # RNG states must be CPU ByteTensors (torch.load with map_location='cuda' moves them)
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'].cpu())
    if torch.cuda.is_available() and 'cuda' in st:
        torch.cuda.set_rng_state_all([t.cpu() for t in st['cuda']])


def fedavg(models, active):
    """Average the client models of the active clients (float tensors) and copy the result to all."""
    states = [models[k].state_dict() for k in active]
    avg = {}
    for name, v in states[0].items():
        if v.dtype.is_floating_point:
            avg[name] = torch.stack([s[name].float() for s in states]).mean(0).to(v.dtype)
        else:
            avg[name] = v.clone()
    for m in models:
        m.load_state_dict(avg)


class SplitFedTrainer:
    """
    K clients with their own client-side model and optimiser, one server-side model.

    client_builder()  returns a fresh client model (same architecture for every client)
    server            the server model
    train_ds          the training dataset (with the training transform)
    shards            from get_split()
    test_loader       for the test accuracy after every round
    rounds            number of rounds
    hooks             an MCHooks (or HookChain); None = plain SplitFed
    best_path         the best model is saved here in the single-client checkpoint format
                      ({'epoch', 'client_state', 'server_state', 'best_acc', ...}), so every attack
                      that loads the single-client checkpoint can load this one
    state_path        per-round training state for resume (None = no resume)
    keep_snapshots    end-of-round snapshots kept for a rollback
    signature         dict that must match for a saved state to be resumed
    """

    def __init__(self, client_builder, server, train_ds, shards, test_loader, rounds, device,
                 hooks=None, best_path=None, state_path=None, seed=0, lr=None, batch_size=None,
                 keep_snapshots=3, signature=None, label='', use_scheduler=True, verbose=True):
        self.K = len(shards)
        self.device = device
        self.rounds = int(rounds)
        self.hooks = hooks or MCHooks()
        self.best_path, self.state_path = best_path, state_path
        self.seed, self.label, self.verbose = seed, label, verbose
        self.lr = lr or Config.LEARNING_RATE
        self.batch_size = batch_size or Config.BATCH_SIZE
        self.keep = max(1, int(keep_snapshots))
        self.train_ds, self.shards, self.test_loader = train_ds, shards, test_loader

        first = client_builder().to(device)
        self.clients = [first] + [copy.deepcopy(first) for _ in range(self.K - 1)]
        self.server = server.to(device)
        self.c_opts = [optim.Adam(c.parameters(), lr=self.lr) for c in self.clients]
        self.s_opt = optim.Adam(self.server.parameters(), lr=self.lr)
        self.use_scheduler = use_scheduler
        # called exactly like the single-client trainer: ReduceLROnPlateau(patience=5, factor=0.5).step(test_acc)
        self.c_scheds = [optim.lr_scheduler.ReduceLROnPlateau(o, patience=5, factor=0.5) for o in self.c_opts]
        self.s_sched = optim.lr_scheduler.ReduceLROnPlateau(self.s_opt, patience=5, factor=0.5)
        self.criterion = nn.CrossEntropyLoss()

        self.active = list(range(self.K))
        self.excluded = {}                     # client -> round in which it was excluded
        self.round = 0                         # last finished round
        self.steps = 0                         # rounds trained, rollbacks included
        self.best_acc, self.best_round = -1.0, 0
        self.history = []                      # one dict per trained round
        self.events = []                       # rollbacks / exclusions
        self.snaps = {}
        self.signature = dict(signature or {}, K=self.K, rounds=self.rounds, seed=seed, hooks=self.hooks.tag(),
                              shard_sizes=[int(len(p)) for p in shards])

    # ---------------------------------------------------------------- state
    def global_client(self):
        """The averaged client model (all active clients hold the same weights after FedAvg)."""
        return self.clients[self.active[0] if self.active else 0]

    def _snapshot(self):
        return {'client': _cpu_state(self.global_client()), 'server': _cpu_state(self.server),
                'c_opts': [copy.deepcopy(o.state_dict()) for o in self.c_opts],
                's_opt': copy.deepcopy(self.s_opt.state_dict()),
                'c_scheds': [s.state_dict() for s in self.c_scheds], 's_sched': self.s_sched.state_dict()}

    def _restore(self, snap):
        for c in self.clients:
            c.load_state_dict(snap['client'])
        self.server.load_state_dict(snap['server'])
        for o, st in zip(self.c_opts, snap['c_opts']):
            o.load_state_dict(st)
        self.s_opt.load_state_dict(snap['s_opt'])
        for s, st in zip(self.c_scheds, snap['c_scheds']):
            s.load_state_dict(st)
        self.s_sched.load_state_dict(snap['s_sched'])

    def _save_state(self):
        if not self.state_path:
            return
        _atomic_save({'signature': self.signature, 'round': self.round, 'steps': self.steps,
                      'active': self.active, 'excluded': self.excluded, 'best_acc': self.best_acc,
                      'best_round': self.best_round, 'history': self.history, 'events': self.events,
                      'now': self._snapshot(), 'snaps': self.snaps, 'rng': _rng_state(),
                      'hooks_state': getattr(self.hooks, 'state_dict', lambda: None)()}, self.state_path)

    def _load_state(self):
        if not self.state_path or not os.path.exists(self.state_path):
            return False
        st = torch.load(self.state_path, map_location='cpu', weights_only=False)
        if st.get('signature') != self.signature:
            print(f"  [resume] {self.state_path} belongs to a different setting; starting fresh")
            return False
        self._restore(st['now'])
        self.round, self.steps = st['round'], st['steps']
        self.active, self.excluded = st['active'], st['excluded']
        self.best_acc, self.best_round = st['best_acc'], st['best_round']
        self.history, self.events, self.snaps = st['history'], st['events'], st['snaps']
        _set_rng_state(st['rng'])
        if st.get('hooks_state') is not None and hasattr(self.hooks, 'load_state_dict'):
            self.hooks.load_state_dict(st['hooks_state'])
        print(f"  [resume] continuing after round {self.round} (best test accuracy so far {self.best_acc:.2f}%)")
        return True

    # ---------------------------------------------------------------- training
    def _loader(self, k, rnd):
        # order fixed by (seed, round, client): a resumed or repeated run sees the same batches
        g = torch.Generator().manual_seed(int(self.seed) * 1_000_003 + rnd * 1_009 + k)
        return DataLoader(Subset(self.train_ds, self.shards[k].tolist()), batch_size=self.batch_size,
                          shuffle=True, drop_last=True, generator=g, num_workers=0,
                          pin_memory=torch.cuda.is_available())

    def _step(self, k, info, x, y):
        c, opt = self.clients[k], self.c_opts[k]
        x, y = self.hooks.batch(info, x, y)
        opt.zero_grad()
        self.s_opt.zero_grad()
        z = c(x)
        zs = self.hooks.smashed(info, z.detach())
        if zs.shape != z.shape:
            raise RuntimeError(f"the smashed hook changed the shape {tuple(z.shape)} -> {tuple(zs.shape)}")
        zs = zs.detach().requires_grad_(True)
        out = self.server(zs)
        loss = self.criterion(out, y)
        loss.backward()
        self.s_opt.step()
        g = self.hooks.grad(info, zs.grad)
        z.backward(g)
        opt.step()
        return loss.item(), int(out.argmax(1).eq(y).sum().item()), int(y.numel())

    def evaluate(self, loader=None):
        c, s = self.global_client(), self.server
        c.eval(); s.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in (loader or self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                correct += s(c(x)).argmax(1).eq(y).sum().item()
                total += y.numel()
        return 100.0 * correct / max(total, 1)

    def train_round(self, rnd, max_batches=None):
        for m in self.clients + [self.server]:
            m.train()
        iters = {k: iter(self._loader(k, rnd)) for k in self.active}
        count = {k: 0 for k in self.active}
        alive = list(self.active)
        loss_sum = correct = seen = nb = 0
        while alive:
            for k in list(alive):
                if max_batches is not None and count[k] >= max_batches:
                    alive.remove(k)
                    continue
                try:
                    x, y = next(iters[k])
                except StopIteration:
                    alive.remove(k)
                    continue
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                info = {'client': k, 'round': rnd, 'batch': count[k], 'labels': y}
                l, c, n = self._step(k, info, x, y)
                loss_sum += l; correct += c; seen += n; nb += 1
                count[k] += 1
        fedavg(self.clients, self.active)
        return {'loss': loss_sum / max(nb, 1), 'train_acc': 100.0 * correct / max(seen, 1), 'batches': nb}

    def train(self, max_batches=None):
        """Train until `rounds` rounds are finished; returns (global client model, server, history)."""
        resumed = self._load_state()
        if not resumed and self.verbose:
            print(f"\n  SplitFed {self.label}: {self.K} client(s), {self.rounds} rounds, hooks: {self.hooks.tag()}")
        while self.round < self.rounds:
            rnd = self.round + 1
            t0 = time.time()
            stats = self.train_round(rnd, max_batches=max_batches)
            acc = self.evaluate()
            if self.use_scheduler:
                for s in self.c_scheds:
                    s.step(acc)
                self.s_sched.step(acc)
            self.round, self.steps = rnd, self.steps + 1
            row = {'step': self.steps, 'round': rnd, 'active': ' '.join(map(str, self.active)),
                   'test_acc': acc, 'lr': self.s_opt.param_groups[0]['lr'], 'seconds': round(time.time() - t0, 1),
                   **stats}
            self.snaps[rnd] = self._snapshot()
            for old in [r for r in self.snaps if r <= rnd - self.keep]:
                del self.snaps[old]

            decision = self.hooks.after_round(self, rnd)
            if decision:
                back = int(decision.get('rollback_to', rnd))
                excl = [int(k) for k in decision.get('exclude', []) if int(k) in self.active]
                for k in excl:
                    self.active.remove(k)
                    self.excluded[k] = rnd
                if not self.active:
                    raise RuntimeError("every client is excluded")
                if back < rnd:
                    if back not in self.snaps:
                        back = min(self.snaps)      # oldest kept snapshot (rounds before 1 are not kept)
                    self._restore(self.snaps[back])
                    self.snaps = {r: v for r, v in self.snaps.items() if r <= back}
                    self.round = back
                    if self.best_round > back:      # the best model came from rounds that were undone
                        self.best_acc, self.best_round = -1.0, 0
                ev = {'step': self.steps, 'round': rnd, 'rollback_to': back, 'exclude': excl,
                      'reason': decision.get('reason', '')}
                self.events.append(ev)
                row.update(rollback_to=back, excluded=' '.join(map(str, excl)))
                if self.verbose:
                    print(f"  round {rnd}: {decision.get('reason', 'defence decision')} -> exclude {excl}, "
                          f"rollback to round {back}")
            self.history.append(row)

            if acc > self.best_acc and not (decision and int(decision.get('rollback_to', rnd)) < rnd):
                self.best_acc, self.best_round = acc, rnd
                if self.best_path:
                    _atomic_save({'epoch': rnd - 1, 'client_state': self.global_client().state_dict(),
                                  'server_state': self.server.state_dict(), 'best_acc': acc,
                                  'dataset': Config.DATASET, 'cut_layer': Config.CUT_LAYER, 'lr': self.lr,
                                  'multiclient': self.signature}, self.best_path)
            if self.verbose:
                print(f"  round {rnd:3d}/{self.rounds} | loss {stats['loss']:.4f} | train {stats['train_acc']:.2f}% "
                      f"| test {acc:.2f}% | clients {row['active']} | {row['seconds']:.0f}s", flush=True)
            self._save_state()
        if self.verbose:
            print(f"  SplitFed finished: best test accuracy {self.best_acc:.2f}% (round {self.best_round})")
        return self.global_client(), self.server, self.history