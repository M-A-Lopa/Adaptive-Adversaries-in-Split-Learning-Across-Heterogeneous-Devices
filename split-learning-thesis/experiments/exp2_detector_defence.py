"""
Experiment 2 -- does the class-conditional detector actually stop the client backdoor?

Experiment 1 showed that poisoned and clean target-class features separate after
training.  Here the detector runs DURING training, on the server side, and the
samples it flags are dropped from training.  We then measure whether the backdoor
(ASR) goes down and what it costs in clean accuracy.

Per run (dataset, model, cut, poison rate, rule):
  1. same client backdoor as the thesis attack: 4x4 white patch in the bottom-right
     corner, label changed to the target class (default 0), poison rate 0 / 1 / 5 / 10 %
  2. split training for --epochs epochs; every epoch the server stores a low-dimensional
     random projection of each sample's smashed data (it already sees the smashed data,
     nothing new is revealed)
  3. from epoch --warmup on, at the end of every epoch, for EVERY class (the server does
     not know the target class):
        - top-10 PCA of that class's features + 2-means split
        - silhouette score of the split
        - the class is "suspicious" if its silhouette is clearly higher than the other
          classes' (above median + k * MAD) and above an absolute floor
        - in a suspicious class, one of the two clusters is dropped, chosen by the rule:
            smaller : the smaller cluster
            tighter : the more compact cluster (a shared trigger makes poisoned features alike)
            crossclass : the cluster whose features lie closer to the OTHER classes' centroids
                      (poisoned images are really other classes' images with a trigger)
            oracle  : the cluster with more truly poisoned samples (upper bound; uses labels
                      a real server does not have)
        - response 'retrain' (default): the poison is found after --warmup epochs, then a FRESH model
          is trained --epochs epochs without the flagged samples (dropping poison from a model that
          already learned the backdoor does not remove it)
        - response 'filter': one model, flagged samples get zero loss weight from epoch --warmup on
  4. clean accuracy and ASR on the test set, detection precision / recall

Rule "none" trains without the detector: it is the undefended baseline.

Run from the project root (see run_exp2.bat):
    python experiments/exp2_detector_defence.py --quick
    python experiments/exp2_detector_defence.py --dataset CIFAR10

Resume: run the same command again.  Finished runs are skipped; an interrupted run
continues from its last finished epoch (model, optimisers, dropped-sample mask and RNG
state are checkpointed every epoch).

Results: <RESULTS_DIR>/exp2_defence/exp2_results_<dataset>.csv  (+ _epochs.csv, + PDF)
"""
import os
import sys
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# reuse the trigger, poisoned dataset, checkpoint helpers and PCA / 2-means from Experiment 1
from exp1_detector_feasibility import (ROOT, PatchTrigger, loader, atomic_save, atomic_csv, _rng_state,
                                       _set_rng_state, clean_acc_and_asr, two_means)


def target_suffix(target):
    """File-name suffix for the target class: none for class 0 (the default), '_t<k>' otherwise."""
    return '' if int(target) == 0 else f"_t{int(target)}"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from sklearn.metrics import silhouette_score

from config import Config
from util.split_env import build_split_models, load_raw_datasets, get_targets, set_seed, get_device

PROJ_DIM = 256          # random projection size for the stored features (keeps memory small)
PCA_Q = 10
SIL_SAMPLE = 2000       # points used to estimate the silhouette score
RULES = ('none', 'smaller', 'tighter', 'crossclass', 'oracle')


class IndexedPoisonedView(Dataset):
    """Like Experiment 1's PoisonedView, but also returns the sample index."""

    def __init__(self, base, poison_idx, target, trigger):
        self.base, self.target, self.trigger = base, int(target), trigger
        self.poison = set(int(i) for i in poison_idx)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        if i in self.poison:
            return self.trigger(x), self.target, 1, i
        return x, int(y), 0, i


# --------------------------------------------------------------------------- #
#  Detector                                                                    #
# --------------------------------------------------------------------------- #
def _pca(x, q=PCA_Q):
    x = x - x.mean(dim=0, keepdim=True)
    q = min(q, x.shape[0] - 1, x.shape[1])
    _, _, v = torch.pca_lowrank(x, q=q, center=False, niter=4)
    return x @ v


def detect(feats, labels, num_classes, rule, poison_flags, args, rng):
    """
    feats: (N, PROJ_DIM) features of this epoch, labels: (N,) labels as the server sees them.
    Returns (drop mask (N,) bool, per-class info list).
    """
    drop = np.zeros(len(labels), dtype=bool)
    info, sils, splits = [], {}, {}
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        if len(idx) < 20:
            continue
        coords = _pca(feats[idx])
        assign = two_means(coords)
        if assign.min() == assign.max():
            continue
        sub = rng.choice(len(idx), size=min(SIL_SAMPLE, len(idx)), replace=False)
        if len(np.unique(assign[sub])) < 2:
            continue
        sils[c] = float(silhouette_score(coords[sub].numpy(), assign[sub]))
        splits[c] = (idx, coords, assign)

    if not sils:
        return drop, info
    centroids = {c: feats[np.where(labels == c)[0]].mean(dim=0) for c in range(num_classes)
                 if (labels == c).any()}
    vals = np.array(list(sils.values()))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) + 1e-6
    thr = max(args.sil_floor, med + args.mad_k * mad)

    for c, s in sils.items():
        idx, coords, assign = splits[c]
        suspicious = s >= thr
        chosen = None
        if suspicious:
            sizes = [(assign == k).sum() for k in (0, 1)]
            if rule == 'smaller':
                chosen = int(np.argmin(sizes))
            elif rule == 'tighter':
                spread = []
                for k in (0, 1):
                    pts = coords[assign == k]
                    spread.append(float((pts - pts.mean(0)).norm(dim=1).mean()))
                chosen = int(np.argmin(spread))
            elif rule == 'crossclass':
                others = torch.stack([v for k, v in centroids.items() if k != c])
                near = []
                for k in (0, 1):
                    pts = feats[idx[assign == k]]
                    near.append(float(torch.cdist(pts, others).min(dim=1).values.mean()) if len(pts) else np.inf)
                chosen = int(np.argmin(near))
            elif rule == 'oracle':
                frac = [poison_flags[idx][assign == k].mean() if sizes[k] else 0.0 for k in (0, 1)]
                chosen = int(np.argmax(frac))
            drop[idx[assign == chosen]] = True
        info.append({'class': c, 'silhouette': s, 'threshold': thr, 'suspicious': bool(suspicious),
                     'dropped': int((assign == chosen).sum()) if chosen is not None else 0})
    return drop, info


# --------------------------------------------------------------------------- #
#  One run                                                                     #
# --------------------------------------------------------------------------- #
def run_one(raw, dataset, model, cut, rate, rule, args, device):
    t0 = time.time()
    set_seed(args.seed)
    targets = get_targets(raw['train_plain'])
    n_train = len(targets)
    rng_poison = np.random.RandomState(args.seed)
    pool = np.where(targets != args.target)[0]
    n_poison = int(round(rate * n_train))
    poison_idx = rng_poison.choice(pool, size=n_poison, replace=False) if n_poison else np.array([], dtype=int)
    poison_flags = np.zeros(n_train, dtype=bool)
    poison_flags[poison_idx] = True
    trigger = PatchTrigger(dataset, size=args.trigger_size)
    view = IndexedPoisonedView(raw['train_aug'], poison_idx, args.target, trigger)
    seen_labels = targets.copy()
    seen_labels[poison_idx] = args.target          # labels the server receives

    client, server = build_split_models(model, cut, dataset, device=device)
    c_opt = optim.Adam(client.parameters(), lr=args.lr)
    s_opt = optim.Adam(server.parameters(), lr=args.lr)
    with torch.no_grad():
        client.eval()
        dim = client(view[0][0].unsqueeze(0).to(device)).flatten(1).shape[1]
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1234)
    proj = (torch.randn(dim, PROJ_DIM, generator=g) / PROJ_DIM ** 0.5).to(device)

    tag = f"{dataset}_{model.lower()}_cut{cut}_rate{rate:g}_{rule}{target_suffix(args.target)}"
    ckpt_path = None if args.quick else os.path.join(args.ckpt_dir, f"run_{tag}.pt")
    signature = {k: getattr(args, k) for k in ('epochs', 'lr', 'batch_size', 'seed', 'target', 'trigger_size',
                                                'warmup', 'sil_floor', 'mad_k', 'response')}
    signature.update(dataset=dataset, model=model, cut=cut, rate=rate, rule=rule)

    # retrain: phase A = `warmup` epochs on all data to find the poison, phase B = a FRESH model trained
    #          `epochs` epochs without the flagged samples (classic detect-then-retrain, Tran et al. 2018)
    # filter : one model; from epoch `warmup` on, flagged samples are dropped while training continues
    retrain = args.response == 'retrain' and rule != 'none'
    total = args.warmup + args.epochs if retrain else args.epochs
    phase = 'A' if retrain else 'B'
    drop = np.zeros(n_train, dtype=bool)
    epoch_log, start = [], 0
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get('signature') == signature:
            client.load_state_dict(ck['client']); server.load_state_dict(ck['server'])
            c_opt.load_state_dict(ck['c_opt']); s_opt.load_state_dict(ck['s_opt'])
            _set_rng_state(ck['rng'])
            drop, epoch_log, start = ck['drop'], ck['epoch_log'], int(ck['epoch'])
            phase = ck.get('phase', phase)
            print(f"    [resume] continuing from epoch {start + 1}/{total}", flush=True)
        else:
            print("    [resume] checkpoint is from different settings; starting this run from epoch 1", flush=True)

    rng_det = np.random.RandomState(args.seed + 7)
    mb = args.max_batches
    for ep in range(start, total):
        if retrain and phase == 'A' and ep == args.warmup:
            # restart from a fresh model; the poison found in phase A stays out of training
            set_seed(args.seed + 99)
            client, server = build_split_models(model, cut, dataset, device=device)
            c_opt = optim.Adam(client.parameters(), lr=args.lr)
            s_opt = optim.Adam(server.parameters(), lr=args.lr)
            phase = 'B'
            print(f"    [retrain] fresh model, training {args.epochs} epochs without {int(drop.sum())} flagged "
                  f"samples", flush=True)
        client.train(); server.train()
        feats = torch.zeros(n_train, PROJ_DIM)
        got = np.zeros(n_train, dtype=bool)
        drop_t = torch.from_numpy(drop)
        run_loss, nb = 0.0, 0
        for b, (x, y, _, idx) in enumerate(loader(view, True, args.batch_size, drop_last=True)):
            if mb and b >= mb:
                break
            x, y = x.to(device), y.to(device)
            keep = (~drop_t[idx]).float().to(device)
            c_opt.zero_grad(); s_opt.zero_grad()
            z = client(x)
            zs = z.detach().requires_grad_(True)
            per = F.cross_entropy(server(zs), y, reduction='none')
            loss = (per * keep).sum() / keep.sum().clamp_min(1.0)
            loss.backward()
            s_opt.step()
            z.backward(zs.grad)
            c_opt.step()
            with torch.no_grad():
                feats[idx] = (zs.detach().flatten(1) @ proj).float().cpu()
            got[idx.numpy()] = True
            run_loss += loss.item(); nb += 1

        row = {'epoch': ep + 1, 'phase': phase, 'loss': run_loss / max(nb, 1), 'dropped': int(drop.sum()),
               'dropped_poison': int((drop & poison_flags).sum()), 'dropped_clean': int((drop & ~poison_flags).sum())}
        detect_now = rule != 'none' and ((retrain and phase == 'A' and ep + 1 == args.warmup) or
                                         (not retrain and ep + 1 >= args.warmup))
        if detect_now:
            ids = np.where(got)[0]
            d_sub, info = detect(feats[ids], seen_labels[ids], Config.NUM_CLASSES, rule, poison_flags[ids], args, rng_det)
            drop = np.zeros(n_train, dtype=bool)
            drop[ids[d_sub]] = True
            tgt = next((i for i in info if i['class'] == args.target), None)
            others = [i['silhouette'] for i in info if i['class'] != args.target]
            row.update({'next_dropped': int(drop.sum()), 'next_dropped_poison': int((drop & poison_flags).sum()),
                        'next_dropped_clean': int((drop & ~poison_flags).sum()),
                        'suspicious_classes': ' '.join(str(i['class']) for i in info if i['suspicious']),
                        'target_silhouette': tgt['silhouette'] if tgt else float('nan'),
                        'other_silhouette_max': max(others) if others else float('nan'),
                        'threshold': info[0]['threshold'] if info else float('nan')})
        epoch_log.append(row)
        extra = ''
        if 'next_dropped' in row:
            extra = (f" | detector: drop {row['next_dropped']} (poison {row['next_dropped_poison']}, "
                     f"clean {row['next_dropped_clean']}) | suspicious [{row['suspicious_classes']}] "
                     f"| target sil {row['target_silhouette']:.3f} vs others max {row['other_silhouette_max']:.3f}")
        ph = f" [{'find poison' if phase == 'A' else 'train'}]" if retrain else ''
        print(f"    epoch {ep + 1}/{total}{ph}  loss {row['loss']:.4f}{extra}", flush=True)
        if ckpt_path:
            atomic_save({'signature': signature, 'epoch': ep + 1, 'phase': phase, 'client': client.state_dict(),
                         'server': server.state_dict(), 'c_opt': c_opt.state_dict(), 's_opt': s_opt.state_dict(),
                         'rng': _rng_state(), 'drop': drop, 'epoch_log': epoch_log}, ckpt_path)

    acc, asr = clean_acc_and_asr(client, server, raw['test_plain'], args.target, trigger, device,
                                 args.batch_size, mb)
    # detection quality of the mask that was active in the last epoch
    last_drop = np.zeros(n_train, dtype=bool) if rule == 'none' else drop
    tp = int((last_drop & poison_flags).sum())
    res = {'dataset': dataset, 'model': model, 'cut': cut, 'poison_rate': rate, 'rule': rule,
           'target': args.target, 'n_poison': int(n_poison), 'clean_acc': acc, 'asr': asr,
           'dropped_last': int(last_drop.sum()), 'dropped_poison_last': tp,
           'dropped_clean_last': int((last_drop & ~poison_flags).sum()),
           'det_precision': tp / last_drop.sum() if last_drop.sum() else float('nan'),
           'det_recall': tp / n_poison if n_poison else float('nan'),
           'clean_removed_pct': 100.0 * (last_drop & ~poison_flags).sum() / max((~poison_flags).sum(), 1),
           'epochs_flagging_target': sum(1 for r in epoch_log
                                         if str(args.target) in str(r.get('suspicious_classes', '')).split()),
           'epochs': args.epochs, 'warmup': args.warmup, 'response': args.response if rule != 'none' else '-',
           'total_epochs': total, 'seed': args.seed,
           'seconds': round(time.time() - t0, 1)}
    print(f"  -> clean acc {acc:.2f}% | ASR {asr:.2f}% | dropped {res['dropped_last']} "
          f"(poison {tp}/{n_poison}, clean {res['dropped_clean_last']})", flush=True)
    for r in epoch_log:
        r.update({'dataset': dataset, 'model': model, 'cut': cut, 'poison_rate': rate, 'rule': rule})
    del client, server
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res, epoch_log, ckpt_path


def main():
    ap = argparse.ArgumentParser(description="Experiment 2: detector in the training loop vs the client backdoor")
    ap.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'MNIST'])
    ap.add_argument('--models', nargs='+', default=['KAGN', 'PyramidCNN'])
    ap.add_argument('--cuts', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.10])
    ap.add_argument('--rules', nargs='+', default=['none', 'smaller', 'crossclass', 'oracle'], choices=RULES,
                    help="none = undefended baseline; oracle = upper bound using the true poison labels")
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--response', default='retrain', choices=['retrain', 'filter'],
                    help="retrain: find poison in `warmup` epochs, then train a fresh model without it; "
                         "filter: keep training one model and drop flagged samples from epoch `warmup` on")
    ap.add_argument('--warmup', type=int, default=3,
                    help="retrain: epochs used to find the poison; filter: first epoch the detector runs")
    ap.add_argument('--sil-floor', type=float, default=0.10, help="minimum silhouette to call a class suspicious")
    ap.add_argument('--mad-k', type=float, default=3.0,
                    help="a class is suspicious if its silhouette > median + k * MAD of all classes")
    ap.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--keep-ckpt', action='store_true')
    ap.add_argument('--quick', action='store_true', help="smoke test: 3 epochs, few batches, one config")
    args = ap.parse_args()

    args.max_batches = None
    if args.quick:
        args.epochs, args.warmup, args.max_batches = 3, 1, 40
        args.models, args.cuts, args.rates, args.rules = args.models[:1], args.cuts[:1], [0.05], ['none', 'crossclass']

    os.chdir(ROOT)
    device = get_device()
    args.out_dir = os.path.join(Config.RESULTS_DIR, 'exp2_defence')
    args.ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    tsuf = target_suffix(args.target) + ('_quick' if args.quick else '')
    out_csv = os.path.join(args.out_dir, f"exp2_results_{args.dataset}{tsuf}.csv")
    ep_csv = os.path.join(args.out_dir, f"exp2_epochs_{args.dataset}{tsuf}.csv")

    raw = load_raw_datasets(args.dataset)
    rows = pd.read_csv(out_csv).to_dict('records') if os.path.exists(out_csv) and not args.quick else []
    ep_rows = pd.read_csv(ep_csv).to_dict('records') if os.path.exists(ep_csv) and not args.quick else []
    done = {(r['model'], int(r['cut']), float(r['poison_rate']), r['rule']) for r in rows}
    todo = [(m, c, r, u) for m in args.models for c in args.cuts for r in args.rates for u in args.rules]
    print(f"[plan] {len(todo)} runs, {sum(t in done for t in todo)} already finished, results -> {out_csv}")

    try:
        for i, (model, cut, rate, rule) in enumerate(todo, 1):
            if (model, cut, rate, rule) in done:
                print(f"[skip {i}/{len(todo)}] {model} cut {cut} rate {rate:g} rule {rule} already finished")
                continue
            print(f"\n=== run {i}/{len(todo)} | {args.dataset} | {model} | cut {cut} | poison {rate:g} | rule {rule} ===",
                  flush=True)
            res, log, ckpt = run_one(raw, args.dataset, model, cut, rate, rule, args, device)
            rows.append(res)
            ep_rows.extend(log)
            atomic_csv(ep_rows, ep_csv)
            atomic_csv(rows, out_csv)         # the run counts as finished only once it is in the CSV
            if ckpt and os.path.exists(ckpt) and not args.keep_ckpt:
                os.remove(ckpt)
    except KeyboardInterrupt:
        print("\n[stopped] Run the same command again to continue: finished runs are skipped and the "
              "interrupted run resumes from its last finished epoch.")
        sys.exit(130)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print("\n" + df[['model', 'cut', 'poison_rate', 'rule', 'clean_acc', 'asr', 'det_precision', 'det_recall',
                     'clean_removed_pct']].to_string(index=False))
    print(f"\nsaved -> {out_csv}")
    try:
        from exp2_report import build_report
        print(f"PDF report -> {build_report(out_csv, ep_csv)}")
    except Exception as exc:
        print(f"[warn] PDF report failed ({type(exc).__name__}: {exc}); "
              f"rebuild with: python experiments/exp2_report.py {out_csv}")


if __name__ == '__main__':
    main()