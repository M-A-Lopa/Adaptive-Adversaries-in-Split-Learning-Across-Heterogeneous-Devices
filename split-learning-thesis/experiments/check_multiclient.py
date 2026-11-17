"""
Step 1 check of the multi-client setup (util/multiclient.py).  Short: about 1-3 minutes on a GPU.

Run from the project root:
    python experiments/check_multiclient.py
    python experiments/check_multiclient.py --dataset MNIST --split noniid
    python experiments/check_multiclient.py --clients 10

It checks, and prints PASS / FAIL for each:
  1. the client split is made, saved, and read back identically (and shows the images per class)
  2. SplitFed training runs and learns (test accuracy above chance after a few short rounds)
  3. the hooks are called for every client and a defence can exclude a client and roll back
  4. resume: a finished run is picked up from its saved state instead of being trained again
Nothing here changes the old single-client results.  Files go to results/mc_check/ and
results/splits/ (the split file is the real one that every later experiment will read).
"""
import argparse
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import Config                                           # noqa: E402
from dataset import DatasetLoader                                   # noqa: E402
from util.split_env import build_split_models, set_seed, get_device  # noqa: E402
from util import multiclient as MC                                  # noqa: E402


class CountingHook(MC.MCHooks):
    """Counts the batches of every client; in round `act_round` asks to exclude `bad` and roll back."""

    def __init__(self, K, bad, act_round):
        self.calls = {k: 0 for k in range(K)}
        self.bad, self.act_round, self.done = bad, act_round, False

    def batch(self, info, x, y):
        self.calls[info['client']] += 1
        return x, y

    def after_round(self, trainer, rnd):
        if rnd == self.act_round and not self.done:
            self.done = True
            return {'rollback_to': rnd - 1, 'exclude': [self.bad], 'reason': 'check: test exclusion'}
        return None

    def tag(self):
        return 'check'

    def state_dict(self):
        return {'calls': self.calls, 'done': self.done}

    def load_state_dict(self, st):
        self.calls, self.done = st['calls'], st['done']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default=Config.DATASET, choices=['CIFAR10', 'MNIST'])
    ap.add_argument('--model', default=Config.MODEL_NAME, choices=['KAGN', 'PyramidCNN'])
    ap.add_argument('--cut', type=int, default=Config.CUT_LAYER, choices=[1, 2, 3, 4])
    ap.add_argument('--clients', type=int, default=None, help='overrides NUM_CLIENTS of config.py')
    ap.add_argument('--split', default=None, choices=list(MC.SPLITS), help='overrides DATA_SPLIT')
    ap.add_argument('--alpha', type=float, default=None, help='overrides NONIID_ALPHA')
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--max-batches', type=int, default=30, help='batches per client per round (short check)')
    a = ap.parse_args()

    Config.DATASET, Config.MODEL_NAME, Config.CUT_LAYER = a.dataset, a.model, a.cut
    s = MC.mc_settings(num_clients=a.clients, split=a.split, alpha=a.alpha)
    if s['num_clients'] < 2:
        sys.exit("NUM_CLIENTS is 1 (the original single-client setup); use --clients 5 to check the multi-client code")
    set_seed(s['seed'])
    device = get_device()
    K = s['num_clients']
    print("=" * 78)
    print(f"  Multi-client check | {a.dataset} | {a.model} | cut {a.cut} | {MC.describe(s)}")
    print("=" * 78)

    dl = DatasetLoader(dataset_name=a.dataset)
    train_ds, test_ds = dl.train_loader.dataset, dl.test_loader.dataset
    test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(test_ds, range(2000)),
                                              batch_size=Config.BATCH_SIZE, shuffle=False)
    results = []

    # 1. split
    print("\n[1] client split")
    shards = MC.get_split(a.dataset, train_ds, s)
    again = MC.get_split(a.dataset, train_ds, s, verbose=False)
    flat = np.concatenate(shards)
    ok = (all(np.array_equal(p, q) for p, q in zip(shards, again)) and len(flat) == len(train_ds)
          and len(np.unique(flat)) == len(flat))
    results.append(('split saved, read back identically, every image in exactly one client', ok))

    # 2-4 training with hooks, exclusion, rollback, resume
    out = os.path.join(Config.RESULTS_DIR, 'mc_check')
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out, exist_ok=True)
    state = os.path.join(out, 'state.pt')
    best = os.path.join(out, 'best.pth')
    bad = s['malicious'][0]

    def builder():
        return build_split_models(a.model, a.cut, a.dataset, device=device)[0]

    def make(hook):
        server = build_split_models(a.model, a.cut, a.dataset, device=device)[1]
        return MC.SplitFedTrainer(builder, server, train_ds, shards, test_loader, a.rounds, device, hooks=hook,
                                  best_path=best, state_path=state, seed=s['seed'], label='check',
                                  signature={'dataset': a.dataset, 'model': a.model, 'cut': a.cut,
                                             'setting': MC.setting_tag(s)})

    print(f"\n[2] SplitFed training: {a.rounds} short rounds ({a.max_batches} batches per client per round)")
    hook = CountingHook(K, bad, act_round=2 if a.rounds >= 2 else 99)
    tr = make(hook)
    tr.train(max_batches=a.max_batches)
    acc = tr.evaluate()
    results.append((f'training learns: test accuracy {acc:.1f}% > 15%', acc > 15.0))

    print("\n[3] hooks, exclusion and rollback")
    every = all(v > 0 for v in hook.calls.values())
    results.append((f'batch hook called for every client {hook.calls}', every))
    if a.rounds >= 2:
        ok = bad not in tr.active and any(e['exclude'] == [bad] for e in tr.events) and tr.round == a.rounds
        results.append((f'client {bad} excluded in round 2, rolled back, training finished '
                        f'({tr.steps} rounds trained for {a.rounds} rounds)', ok))
    ck = torch.load(best, map_location='cpu', weights_only=False)
    results.append(('best model saved in the single-client checkpoint format',
                    {'client_state', 'server_state', 'best_acc', 'epoch'} <= set(ck)))

    print("\n[4] resume")
    tr2 = make(CountingHook(K, bad, act_round=2 if a.rounds >= 2 else 99))
    tr2.train(max_batches=a.max_batches)
    same = tr2.steps == tr.steps and abs(tr2.evaluate() - acc) < 1e-6
    results.append(('finished run resumed from its saved state, nothing trained again', same))

    print("\n" + "=" * 78)
    for text, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {text}")
    print("=" * 78)
    print("  All checks passed." if all(ok for _, ok in results) else "  Some checks failed: send the output above.")
    shutil.rmtree(out, ignore_errors=True)


if __name__ == '__main__':
    main()