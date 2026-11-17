"""
Experiment 1 -- detector feasibility.

Question: inside the backdoor target class, do poisoned samples separate from
clean samples in the smashed-data (cut-layer) feature space?  If yes, a
class-conditional detector (spectral signature) can find them, which is the
core assumption of the proposed defence.

For every (model, cut, poison rate):
  1. train a vanilla split model on CIFAR-10/MNIST where a fraction of the
     training set carries a BadNets patch trigger relabelled to the target class
  2. check the backdoor was actually learned (clean accuracy + ASR on test)
  3. collect the smashed data of every training sample labelled as the target
     class (clean target-class images + poisoned images)
  4. spectral signature score (Tran et al., NeurIPS 2018): squared projection of
     the centred features on their top singular vector
  5. report AUC (poisoned vs clean) and recall when removing the top 1.5 x n_poison

No existing file is modified.  Place this file in <project>/experiments/ and run
from the project root, e.g.

    python experiments/exp1_detector_feasibility.py --quick            # smoke test
    python experiments/exp1_detector_feasibility.py                    # 12 runs, CIFAR-10
    python experiments/exp1_detector_feasibility.py --dataset MNIST

Results: <RESULTS_DIR>/exp1_detector/exp1_results_<dataset>.csv
"""
import os
import sys
import glob
import time
import argparse


def _find_root(start):
    d = start
    while True:
        if os.path.exists(os.path.join(d, 'config.py')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("config.py not found above " + start)
        d = parent


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = _find_root(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from config import Config
from util.split_env import (NORM_STATS, build_split_models, load_raw_datasets, get_targets,
                            set_seed, get_device, split_step)


# --------------------------------------------------------------------------- #
#  Trigger                                                                     #
# --------------------------------------------------------------------------- #
class PatchTrigger:
    """
    Same trigger as the thesis client backdoor (all_attacks BackdoorPoisonAttack,
    mode='client', apply_trigger_patch): a 4x4 square of value 1.0 (white in [0,1]
    pixel space) in the very bottom-right corner.  Applied here to the already
    normalised tensor, so white is written as (1 - mean) / std per channel, which
    equals denormalise -> set 1.0 -> normalise in the attack code.
    """

    def __init__(self, dataset, size=4, margin=0):
        mean, std = NORM_STATS[dataset]
        white = [(1.0 - m) / s for m, s in zip(mean, std)]
        self.white = torch.tensor(white, dtype=torch.float32).view(-1, 1, 1)
        self.size, self.margin = size, margin

    def __call__(self, x):
        x = x.clone()
        s, m = self.size, self.margin
        h, w = x.shape[-2], x.shape[-1]
        x[..., h - m - s:h - m, w - m - s:w - m] = self.white.to(x.device)
        return x


class PoisonedView(Dataset):
    """Wraps a dataset; samples in poison_idx get the trigger and the target label."""

    def __init__(self, base, poison_idx, target, trigger):
        self.base, self.target, self.trigger = base, int(target), trigger
        self.poison = set(int(i) for i in poison_idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if i in self.poison:
            return self.trigger(x), self.target, 1
        return x, int(y), 0


def loader(ds, shuffle, batch_size, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available(), drop_last=drop_last)


# --------------------------------------------------------------------------- #
#  Training and evaluation                                                     #
# --------------------------------------------------------------------------- #
def _rng_state():
    s = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()}
    if torch.cuda.is_available():
        s['cuda'] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s):
    torch.set_rng_state(s['torch'].cpu())
    np.random.set_state(s['numpy'])
    if torch.cuda.is_available() and 'cuda' in s:
        torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda']])


def atomic_save(obj, path):
    """Write to a temp file first, then rename, so a crash never leaves a half-written checkpoint."""
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(client, server, train_loader, epochs, device, lr, max_batches=None, ckpt_path=None, signature=None):
    """
    Split training with per-epoch checkpoints.  If ckpt_path holds a checkpoint from the same
    run (same signature), training continues from the epoch after the last finished one.
    """
    c_opt = optim.Adam(client.parameters(), lr=lr)
    s_opt = optim.Adam(server.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    start = 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get('signature') == signature:
            client.load_state_dict(ck['client'])
            server.load_state_dict(ck['server'])
            c_opt.load_state_dict(ck['c_opt'])
            s_opt.load_state_dict(ck['s_opt'])
            _set_rng_state(ck['rng'])
            start = int(ck['epoch'])
            print(f"    [resume] continuing from epoch {start + 1}/{epochs} ({ckpt_path})", flush=True)
        else:
            print(f"    [resume] {ckpt_path} is from different settings; starting this run from epoch 1",
                  flush=True)
    for ep in range(start, epochs):
        client.train(); server.train()
        run, n = 0.0, 0
        for b, (x, y, _) in enumerate(train_loader):
            if max_batches and b >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            loss, _ = split_step(client, server, c_opt, s_opt, crit, x, y)
            run += loss; n += 1
        print(f"    epoch {ep + 1}/{epochs}  loss {run / max(n, 1):.4f}", flush=True)
        if ckpt_path:
            atomic_save({'signature': signature, 'epoch': ep + 1, 'client': client.state_dict(),
                         'server': server.state_dict(), 'c_opt': c_opt.state_dict(),
                         's_opt': s_opt.state_dict(), 'rng': _rng_state()}, ckpt_path)


@torch.no_grad()
def clean_acc_and_asr(client, server, test_ds, target, trigger, device, batch_size, max_batches=None):
    client.eval(); server.eval()
    correct = total = hit = trig_total = 0
    for b, (x, y) in enumerate(loader(test_ds, False, batch_size)):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        correct += server(client(x)).argmax(1).eq(y).sum().item()
        total += y.numel()
        keep = y != target
        if keep.any():
            xt = trigger(x[keep])
            hit += server(client(xt)).argmax(1).eq(target).sum().item()
            trig_total += int(keep.sum())
    return 100.0 * correct / max(total, 1), 100.0 * hit / max(trig_total, 1)


@torch.no_grad()
def collect_target_features(client, view, target, device, batch_size, max_batches=None):
    """Smashed data of every training sample whose (possibly poisoned) label is the target."""
    client.eval()
    feats, flags = [], []
    for b, (x, y, p) in enumerate(loader(view, False, batch_size)):
        if max_batches and b >= max_batches:
            break
        keep = y == target
        if not keep.any():
            continue
        z = client(x[keep].to(device)).flatten(1).float()
        feats.append(z)
        flags.append(p[keep])
    return torch.cat(feats), torch.cat(flags).numpy()


# --------------------------------------------------------------------------- #
#  Spectral signature                                                          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pca_coords(feats, q=10):
    """Centred features projected on their top-q principal directions, shape (n, q)."""
    centred = feats - feats.mean(dim=0, keepdim=True)
    q = min(q, centred.shape[0], centred.shape[1])
    try:
        _, _, v = torch.pca_lowrank(centred, q=q, center=False, niter=4)
    except RuntimeError:
        # fall back to CPU if the GPU runs out of memory
        centred = centred.cpu()
        _, _, v = torch.pca_lowrank(centred, q=q, center=False, niter=4)
    return (centred @ v).cpu()


@torch.no_grad()
def two_means(x, iters=50, seed=0):
    """k-means with k = 2 (activation clustering, Chen et al. 2018). Returns 0/1 assignment."""
    g = torch.Generator().manual_seed(seed)
    first = x[torch.randint(len(x), (1,), generator=g)]
    second = x[(x - first).norm(dim=1).argmax()].unsqueeze(0)
    centres = torch.cat([first, second])
    assign = torch.zeros(len(x), dtype=torch.long)
    for _ in range(iters):
        new = torch.cdist(x, centres).argmin(dim=1)
        if torch.equal(new, assign) and _ > 0:
            break
        assign = new
        for k in (0, 1):
            if (assign == k).any():
                centres[k] = x[assign == k].mean(dim=0)
    return assign.numpy()


def detector_scores(feats, flags, coords=None):
    """
    Three readings of the same features:
      ss_auc        classic spectral signature: squared projection on the top direction.
                    Assumes poison is a minority of the class; breaks when it is not.
      sep_auc       how well the top direction separates the two groups, whichever side
                    the poison falls on: max(AUC, 1 - AUC) of the signed projection.
      ac_*          2-means on the top-10 PCA coordinates, flagging the smaller cluster
                    (what a deployed detector would actually do).
    """
    coords = pca_coords(feats, q=10) if coords is None else coords
    proj = coords[:, 0].numpy()
    has_both = 0 < flags.sum() < len(flags)
    out = {'ss_auc': float('nan'), 'sep_auc': float('nan')}
    if has_both:
        out['ss_auc'] = float(roc_auc_score(flags, proj ** 2))
        a = float(roc_auc_score(flags, proj))
        out['sep_auc'] = max(a, 1.0 - a)
    out['ss_recall_at_1.5x'] = recall_at(proj ** 2, flags)

    assign = two_means(coords)
    small = 0 if (assign == 0).sum() <= (assign == 1).sum() else 1
    flagged = assign == small
    tp = int((flagged & (flags == 1)).sum())
    out['ac_flagged'] = int(flagged.sum())
    out['ac_precision'] = tp / max(int(flagged.sum()), 1)
    out['ac_recall'] = tp / max(int(flags.sum()), 1)
    # cluster purity: does the 2-means split follow poisoned / clean at all (either side)?
    agree = float(((assign == 1) == (flags == 1)).mean())
    out['ac_split_match'] = max(agree, 1.0 - agree)
    return out


def recall_at(scores, flags, factor=1.5):
    n_poison = int(flags.sum())
    if n_poison == 0:
        return float('nan')
    k = min(len(scores), int(round(factor * n_poison)))
    top = np.argsort(-scores)[:k]
    return float(flags[top].sum() / n_poison)


# --------------------------------------------------------------------------- #
#  One run                                                                     #
# --------------------------------------------------------------------------- #
def run_one(raw, dataset, model, cut, rate, args, device):
    t0 = time.time()
    set_seed(args.seed)
    targets = get_targets(raw['train_plain'])
    rng = np.random.RandomState(args.seed)
    pool = np.where(targets != args.target)[0]
    n_poison = int(round(rate * len(targets)))
    poison_idx = rng.choice(pool, size=n_poison, replace=False)
    trigger = PatchTrigger(dataset, size=args.trigger_size)

    train_view = PoisonedView(raw['train_aug'], poison_idx, args.target, trigger)
    plain_view = PoisonedView(raw['train_plain'], poison_idx, args.target, trigger)
    mb = args.max_batches

    client, server = build_split_models(model, cut, dataset, device=device)
    tag = f"{dataset}_{model.lower()}_cut{cut}_rate{rate:g}"
    ckpt_path = None if args.quick else os.path.join(args.ckpt_dir, f"run_{tag}.pt")
    signature = {'dataset': dataset, 'model': model, 'cut': cut, 'rate': rate, 'epochs': args.epochs,
                 'lr': args.lr, 'batch_size': args.batch_size, 'seed': args.seed, 'target': args.target,
                 'trigger_size': args.trigger_size}
    train(client, server, loader(train_view, True, args.batch_size, drop_last=True),
          args.epochs, device, args.lr, mb, ckpt_path=ckpt_path, signature=signature)
    acc, asr = clean_acc_and_asr(client, server, raw['test_plain'], args.target, trigger, device,
                                 args.batch_size, mb)

    feats, flags = collect_target_features(client, plain_view, args.target, device, args.batch_size, mb)
    coords = pca_coords(feats, q=10)
    det = detector_scores(feats, flags, coords)
    # always keep the small PCA view (n x 10) for the PDF plots
    np.savez_compressed(os.path.join(args.out_dir, f"pca_{dataset}_{model.lower()}_cut{cut}_rate{rate:g}.npz"),
                        coords=coords.numpy().astype(np.float32), poisoned=flags)

    row = {'dataset': dataset, 'model': model, 'cut': cut, 'poison_rate': rate,
           'n_poison_collected': int(flags.sum()), 'n_clean_target': int(len(flags) - flags.sum()),
           'poison_frac_in_target_class': float(flags.mean()), 'feat_dim': int(feats.shape[1]),
           'clean_acc': acc, 'asr': asr, **det,
           'epochs': args.epochs, 'seed': args.seed, 'seconds': round(time.time() - t0, 1)}
    print(f"  -> clean acc {acc:.2f}% | ASR {asr:.2f}% | poison share in target class "
          f"{row['poison_frac_in_target_class']:.3f}\n"
          f"     spectral AUC {det['ss_auc']:.3f} | separability AUC {det['sep_auc']:.3f} | "
          f"2-means precision {det['ac_precision']:.3f} recall {det['ac_recall']:.3f}", flush=True)
    if args.save_feats:
        out = os.path.join(args.out_dir, f"feats_{dataset}_{model.lower()}_cut{cut}_rate{rate:g}.npz")
        np.savez_compressed(out, feats=feats.cpu().numpy().astype(np.float16), poisoned=flags)
    del client, server, feats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    row['_ckpt'] = ckpt_path
    return row


def atomic_csv(rows, path):
    """Save the CSV through a temp file; retry if the file is open in another program (e.g. Excel)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    pd.DataFrame(rows).to_csv(tmp, index=False)
    for _ in range(30):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            print(f"  [warn] {path} is open in another program; close it (retrying in 2 s)", flush=True)
            time.sleep(2)
    print(f"  [warn] could not write {path}; latest results left in {tmp}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Experiment 1: spectral-signature detector feasibility")
    ap.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'MNIST'])
    ap.add_argument('--models', nargs='+', default=['KAGN', 'PyramidCNN'])
    ap.add_argument('--cuts', type=int, nargs='+', default=[1, 3])
    ap.add_argument('--rates', type=float, nargs='+', default=[0.01, 0.05, 0.10])
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=4,
                    help="side of the square trigger; 4 matches the thesis client backdoor (patch_size=4)")
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-feats', action='store_true', help="also save target-class features (.npz)")
    ap.add_argument('--quick', action='store_true', help="1 epoch, few batches, one config")
    ap.add_argument('--fresh', action='store_true',
                    help="ignore earlier results and checkpoints and start everything again")
    ap.add_argument('--keep-ckpt', action='store_true',
                    help="keep each run's training checkpoint after the run is finished")
    args = ap.parse_args()

    args.max_batches = None
    if args.quick:
        args.epochs, args.max_batches = 1, 20
        args.models, args.cuts, args.rates = args.models[:1], args.cuts[:1], [0.10]

    os.chdir(ROOT)
    device = get_device()
    args.out_dir = os.path.join(Config.RESULTS_DIR, 'exp1_detector')
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"exp1_results_{args.dataset}{'_quick' if args.quick else ''}.csv")
    args.ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.fresh and not args.quick:
        if os.path.exists(out_csv):
            os.replace(out_csv, out_csv + '.old')
            print(f"[fresh] earlier results moved to {out_csv}.old")
        for f in glob.glob(os.path.join(args.ckpt_dir, f"run_{args.dataset}_*.pt")):
            os.remove(f)

    raw = load_raw_datasets(args.dataset)
    rows = pd.read_csv(out_csv).to_dict('records') if os.path.exists(out_csv) and not args.quick else []
    done = {(r['model'], int(r['cut']), float(r['poison_rate'])) for r in rows}
    todo = [(m, c, r) for m in args.models for c in args.cuts for r in args.rates]
    print(f"[plan] {len(todo)} runs, {sum(t in done for t in todo)} already finished, results -> {out_csv}")

    try:
        for i, (model, cut, rate) in enumerate(todo, 1):
            if (model, cut, rate) in done:
                print(f"[skip {i}/{len(todo)}] {model} cut {cut} rate {rate:g} already finished")
                continue
            print(f"\n=== run {i}/{len(todo)} | {args.dataset} | {model} | cut {cut} | poison rate {rate:g} ===",
                  flush=True)
            row = run_one(raw, args.dataset, model, cut, rate, args, device)
            ckpt = row.pop('_ckpt')
            rows.append(row)
            atomic_csv(rows, out_csv)          # the run counts as finished only once it is in the CSV
            if ckpt and os.path.exists(ckpt) and not args.keep_ckpt:
                os.remove(ckpt)
    except KeyboardInterrupt:
        print("\n[stopped] Run the same command again to continue: finished runs are skipped and the "
              "interrupted run resumes from its last finished epoch.")
        sys.exit(130)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print("\n" + df[['model', 'cut', 'poison_rate', 'poison_frac_in_target_class', 'clean_acc', 'asr',
                     'ss_auc', 'sep_auc', 'ac_precision', 'ac_recall']].to_string(index=False))
    print(f"\nsaved -> {out_csv}")
    print("Reading: sep_auc >= 0.9 -> poisoned and clean target-class features separate (idea works);")
    print("         sep_auc < 0.7 -> they do not.  ss_auc is only meaningful while poison is a minority")
    print("         of the target class.  ac_precision/recall = what a deployed 2-means detector would catch.")
    print("Only trust rows with high ASR (e.g. > 80%): if the backdoor was not learned, there is nothing to detect.")

    try:
        from exp1_report import build_report
        pdf = build_report(out_csv, args.out_dir)
        print(f"PDF report -> {pdf}")
    except Exception as exc:  # the CSV is already saved; the report can be rebuilt later
        print(f"[warn] PDF report failed ({type(exc).__name__}: {exc}); "
              f"rebuild with: python experiments/exp1_report.py {out_csv}")


if __name__ == '__main__':
    main()