"""
Experiment 3 -- multi-client split learning: find the malicious CLIENT, exclude it, roll back.

Experiment 2 showed that removing poisoned SAMPLES is not enough: even with 91 % of the
poison removed the backdoor survived, and the server would have to learn thousands of
per-sample flags.  Here the server looks for the client instead of the samples.

Setup (all simulated on one machine)
  * --clients (5) clients, the training set split IID into equal shards
  * one malicious client; from round --poison-start (3) it stamps the thesis trigger
    (4x4 white patch, bottom-right) on part of its shard and relabels it to the target class.
    Poison rates are given as a share of the WHOLE training set (0 / 1 / 5 / 10 %), so they
    compare directly with Experiments 1-2 and with the single-client thesis attack
  * SplitFed-style training: every client has its own client-side model, all share the
    server-side model; batches of the clients are interleaved; client models are averaged
    (FedAvg) at the end of every round.  One round = one pass over all active shards

Detector (server side, end of every round from --detect-from on)
  The server already receives every sample's smashed data, its label and which client sent
  it.  For every class c and client k it compares the client's mean feature with the mean
  feature of the other clients for the same class:

      t[c,k] = || mean_k - median of the other clients' means ||
               / (within-class std * sqrt(1/n_k + 1/n_others))

  With IID data all clients should look alike.  A client whose poisoned samples sit in the
  target class shifts that class mean.  Client score = max over classes of t; the score is
  turned into a robust z-score against all (class, client) pairs of the round.  A client is
  flagged if z >= --z-thr.  Only per-client, per-class sums are needed, so in the final
  defence this runs cheaply inside the two-server secret-shared computation and reveals
  nothing but the flagged client's id.

Response
  flagged client is excluded from all later rounds, and the model is rolled back to the
  checkpoint from before the first round in which that client looked suspicious
  (walking back while its z >= --z-soft); training then continues to --rounds rounds.

Methods
  none       : no defence; the detector only monitors (gives the undefended ASR and the signal)
  detect     : detector + exclusion + rollback, flag when one round's z >= --z-thr
  detect_cum : same, but also flag when the cumulative score (CUSUM) reaches --cusum-h:
               S_k <- max(0, S_k + z_k - --cusum-drift) every round.  Catches a client that is
               only slightly unusual but in every round (low poison).  Tuned offline on the
               saved z-scores of the 'none' runs: drift 4, h 4 flagged the malicious client in all
               48 poisoned configurations and no honest client in any run
  detect_both: detect_cum OR a second cumulative score that compares every client with ITSELF:
               change_k = z_k(this round) - z_k(first checked round),
               B_k <- max(0, B_k + change_k - --self-drift), flag when B_k >= --self-h.
               With non-IID data an honest client is unusual by a roughly constant amount, so its
               z hardly changes; a client that starts poisoning jumps.  The baselines and both sums
               restart after every rollback (the set of compared clients has changed).
               Limit: a client that poisons from its first round has no clean baseline
  oracle : the malicious client is removed exactly when it starts poisoning (upper bound)

Variants (each writes its own result files, named by --tag)
  --noniid 0.5 : label-skew non-IID shards (Dirichlet 0.5) instead of IID
  --clients 10 : more clients
After a rollback every client gets its optimiser state from that round back (use --reset-opt
for the earlier behaviour, where the optimisers restarted from zero).

Run from the project root (see run_exp3.bat):
    python experiments/exp3_client_defence.py --quick
    python experiments/exp3_client_defence.py --dataset CIFAR10

Resume: run the same command again.  Finished runs are skipped; an interrupted run
continues from its last finished round.

Results: <RESULTS_DIR>/exp3_client/exp3_results_<dataset>.csv  (+ _rounds.csv, + PDF)
"""
import os
import sys
import copy
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# reuse the trigger, checkpoint helpers and test-set evaluation from Experiment 1
from exp1_detector_feasibility import (ROOT, PatchTrigger, atomic_save, atomic_csv, _rng_state,
                                       _set_rng_state, clean_acc_and_asr)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from config import Config
from util.split_env import build_split_models, load_raw_datasets, get_targets, set_seed, get_device

PROJ_DIM = 256          # random projection of the smashed data kept for the detector
MIN_N = 20              # minimum samples of a (class, client) pair to be scored
METHODS = ('none', 'detect', 'detect_cum', 'detect_both', 'oracle')


class ClientView(Dataset):
    """One client's shard.  While `active` is True its poison set carries the trigger and target label."""

    def __init__(self, base, shard_idx, poison_idx, target, trigger):
        self.base, self.idx = base, np.asarray(shard_idx)
        self.poison = set(int(i) for i in poison_idx)
        self.target, self.trigger, self.active = int(target), trigger, False

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = int(self.idx[j])
        x, y = self.base[i]
        if self.active and i in self.poison:
            return self.trigger(x), self.target, 1, i
        return x, int(y), 0, i


def make_shards(targets, K, rng, alpha=None, min_size=0):
    """
    Split the training indices among K clients.
      alpha None : IID, equal shards (a random permutation cut into K parts)
      alpha > 0  : label-skew non-IID.  For every class the samples are divided among the clients
                   with proportions drawn from Dirichlet(alpha); small alpha = a client holds mostly
                   a few classes.  Redrawn until every client has at least min_size samples.
    """
    n = len(targets)
    if alpha is None:
        return np.array_split(rng.permutation(n), K)
    for _ in range(200):
        parts = [[] for _ in range(K)]
        for c in np.unique(targets):
            idx = rng.permutation(np.where(targets == c)[0])
            cuts = (np.cumsum(rng.dirichlet([alpha] * K))[:-1] * len(idx)).astype(int)
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [rng.permutation(np.asarray(p, dtype=np.int64)) for p in parts]
    raise RuntimeError(f"could not draw a Dirichlet({alpha}) split with at least {min_size} samples per client")


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def fedavg(bottoms, members):
    """Average the client-side models of `members` and load the result into every client model."""
    states = [bottoms[k].state_dict() for k in members]
    avg = {}
    for name, v in states[0].items():
        if v.dtype.is_floating_point:
            avg[name] = torch.stack([s[name] for s in states]).mean(0)
        else:
            avg[name] = v.clone()
    for b in bottoms:
        b.load_state_dict(avg)


# --------------------------------------------------------------------------- #
#  Detector                                                                    #
# --------------------------------------------------------------------------- #
def client_scores(feats, labels, clients, num_classes, active):
    """
    feats (n, P) torch cpu, labels / clients (n,) numpy for the samples seen this round.
    Returns {client: {'score': max_c t, 'cls': argmax class, 'z': robust z}} for the active clients.
    """
    t = {}
    for c in range(num_classes):
        mc = labels == c
        if mc.sum() < 3 * MIN_N:
            continue
        xc, kc = feats[mc], clients[mc]
        ks = [k for k in active if (kc == k).sum() >= MIN_N]
        if len(ks) < 3:                      # need other clients to compare against
            continue
        keep = np.isin(kc, ks)
        xc, kc = xc[keep], kc[keep]
        n_tot = len(kc)
        sums = {k: xc[kc == k].sum(dim=0) for k in ks}
        ns = {k: int((kc == k).sum()) for k in ks}
        within = sum(float(((xc[kc == k] - sums[k] / ns[k]) ** 2).sum()) for k in ks) / max(n_tot - len(ks), 1)
        means = {k: sums[k] / ns[k] for k in ks}
        for k in ks:
            rest_n = n_tot - ns[k]
            # reference = coordinate-wise MEDIAN of the other clients' means, so that one poisoned
            # client cannot drag the reference and make the honest clients look unusual
            ref = torch.quantile(torch.stack([means[j] for j in ks if j != k]), 0.5, dim=0)
            d = float((means[k] - ref).norm())
            se = (within * (1.0 / ns[k] + 1.0 / rest_n)) ** 0.5
            t[(c, k)] = d / max(se, 1e-12)
    out = {}
    if not t:
        return out
    vals = np.array(list(t.values()))
    med = float(np.median(vals))
    mad = 1.4826 * float(np.median(np.abs(vals - med))) + 1e-9
    for k in active:
        mine = {c: v for (c, kk), v in t.items() if kk == k}
        if not mine:
            continue
        cls = max(mine, key=mine.get)
        out[k] = {'score': mine[cls], 'cls': int(cls), 'z': (mine[cls] - med) / mad}
    return out


# --------------------------------------------------------------------------- #
#  One run                                                                     #
# --------------------------------------------------------------------------- #
def run_one(raw, dataset, model, cut, rate, method, args, device):
    t0 = time.time()
    set_seed(args.seed)
    K, mal, R = args.clients, args.malicious, args.rounds
    targets = get_targets(raw['train_plain'])
    n_train = len(targets)
    rng = np.random.RandomState(args.seed)
    shards = make_shards(targets, K, rng, args.noniid, min_size=4 * args.batch_size)
    client_of = np.zeros(n_train, dtype=np.int64)
    for k, s in enumerate(shards):
        client_of[s] = k

    pool = shards[mal][targets[shards[mal]] != args.target]
    n_poison = min(int(round(rate * n_train)), len(pool))
    poison_idx = rng.choice(pool, size=n_poison, replace=False) if n_poison else np.array([], dtype=int)
    trigger = PatchTrigger(dataset, size=args.trigger_size)
    views = [ClientView(raw['train_aug'], shards[k], poison_idx if k == mal else [], args.target, trigger)
             for k in range(K)]

    client, server = build_split_models(model, cut, dataset, device=device)
    bottoms = [client] + [copy.deepcopy(client) for _ in range(K - 1)]
    b_opts = [optim.Adam(b.parameters(), lr=args.lr) for b in bottoms]
    s_opt = optim.Adam(server.parameters(), lr=args.lr)
    with torch.no_grad():
        client.eval()
        dim = client(views[0][0][0].unsqueeze(0).to(device)).flatten(1).shape[1]
    g = torch.Generator(device='cpu').manual_seed(args.seed + 1234)
    proj = (torch.randn(dim, PROJ_DIM, generator=g) / PROJ_DIM ** 0.5).to(device)

    tag = f"{dataset}_{model.lower()}_cut{cut}_rate{rate:g}_{method}{getattr(args, 'fsuf', '')}"
    ckpt_path = None if args.quick else os.path.join(args.ckpt_dir, f"run_{tag}.pt")
    signature = {k: getattr(args, k) for k in ('rounds', 'clients', 'malicious', 'poison_start', 'detect_from',
                                                'z_thr', 'z_soft', 't_floor', 'keep', 'cusum_drift', 'cusum_h',
                                                'lr', 'batch_size', 'seed', 'noniid', 'reset_opt',
                                                'target', 'trigger_size')}
    signature.update(dataset=dataset, model=model, cut=cut, rate=rate, method=method)
    if method == 'detect_both':
        signature.update(self_drift=args.self_drift, self_h=args.self_h)

    # state that survives a restart
    st = {'round': 0, 'step': 0, 'excluded': [], 'flags': [], 'zhist': {}, 'log': [],
          'snaps': {0: {'bottom': cpu_state(bottoms[0]), 'server': cpu_state(server),
                        's_opt': copy.deepcopy(s_opt.state_dict()),
                        'b_opts': [copy.deepcopy(o.state_dict()) for o in b_opts]}}}
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get('signature') == signature:
            for b, sd in zip(bottoms, ck['bottoms']):
                b.load_state_dict(sd)
            for o, sd in zip(b_opts, ck['b_opts']):
                o.load_state_dict(sd)
            server.load_state_dict(ck['server']); s_opt.load_state_dict(ck['s_opt'])
            _set_rng_state(ck['rng'])
            st = ck['st']
            print(f"    [resume] continuing after round {st['round']}/{R} (step {st['step']})", flush=True)
        else:
            print("    [resume] checkpoint is from different settings; starting this run again", flush=True)

    mb = args.max_batches
    while st['round'] < R:
        rnd = st['round'] + 1
        st['step'] += 1
        poisoning = rnd >= args.poison_start and n_poison > 0
        if method == 'oracle' and rnd >= args.poison_start and n_poison > 0 and mal not in st['excluded']:
            st['excluded'].append(mal)
            st['flags'].append({'client': mal, 'round': rnd, 'z': float('nan'), 'rollback_to': rnd - 1})
        active = [k for k in range(K) if k not in st['excluded']]
        views[mal].active = poisoning and mal in active

        for m in bottoms + [server]:
            m.train()
        feats = torch.zeros(n_train, PROJ_DIM)
        got = np.zeros(n_train, dtype=bool)
        seen_label = np.full(n_train, -1, dtype=np.int64)
        iters = {k: iter(DataLoader(views[k], batch_size=args.batch_size, shuffle=True, num_workers=0,
                                    drop_last=True)) for k in active}
        alive, nb, run_loss, counts = list(active), 0, 0.0, {k: 0 for k in active}
        while alive:
            for k in list(alive):
                if mb and counts[k] >= mb:
                    alive.remove(k)
                    continue
                try:
                    x, y, _, idx = next(iters[k])
                except StopIteration:
                    alive.remove(k)
                    continue
                counts[k] += 1
                x, y = x.to(device), y.to(device)
                b_opts[k].zero_grad(); s_opt.zero_grad()
                z = bottoms[k](x)
                zs = z.detach().requires_grad_(True)
                loss = F.cross_entropy(server(zs), y)
                loss.backward()
                s_opt.step()
                z.backward(zs.grad)
                b_opts[k].step()
                with torch.no_grad():
                    feats[idx] = (zs.detach().flatten(1) @ proj).float().cpu()
                ii = idx.numpy()
                got[ii] = True
                seen_label[ii] = y.cpu().numpy()
                run_loss += loss.item(); nb += 1
        fedavg(bottoms, active)

        row = {'step': st['step'], 'round': rnd, 'loss': run_loss / max(nb, 1), 'poisoning': bool(views[mal].active),
               'active': ' '.join(str(k) for k in active), 'flagged': '', 'rollback_to': ''}
        flagged = None
        if rnd >= args.detect_from:
            ids = np.where(got)[0]
            sc = client_scores(feats[ids], seen_label[ids], client_of[ids], Config.NUM_CLASSES, active)
            for k in range(K):
                row[f'score_c{k}'] = sc[k]['score'] if k in sc else float('nan')
                row[f'z_c{k}'] = sc[k]['z'] if k in sc else float('nan')
                row[f'cls_c{k}'] = sc[k]['cls'] if k in sc else -1
                st['zhist'].setdefault(k, {})[rnd] = sc[k]['z'] if k in sc else float('nan')
            cand = [(v['z'], k) for k, v in sc.items() if v['z'] >= args.z_thr and v['score'] >= args.t_floor]
            if cand and method == 'detect':
                flagged = max(cand)[1]
            row['would_flag'] = ' '.join(str(k) for _, k in sorted(cand, reverse=True))
            # cumulative score (CUSUM): S_k <- max(0, S_k + z_k - drift).  A client that is only a little
            # unusual, but in every round, builds up S; an honest client's z goes up and down, so S stays near 0.
            cus, czero = st.setdefault('cusum', {}), st.setdefault('cusum_zero', {})
            cum_cand = []
            for k, v in sc.items():
                cus[k] = max(0.0, cus.get(k, 0.0) + v['z'] - args.cusum_drift)
                if cus[k] == 0.0:
                    czero[k] = rnd
                if (v['z'] >= args.z_thr or cus[k] >= args.cusum_h) and v['score'] >= args.t_floor:
                    cum_cand.append((cus[k] + v['z'], k))
            for k in range(K):
                row[f'cusum_c{k}'] = cus.get(k, float('nan')) if k in sc else float('nan')
            row['would_flag_cum'] = ' '.join(str(k) for _, k in sorted(cum_cand, reverse=True))
            if cum_cand and method == 'detect_cum':
                flagged = max(cum_cand)[1]
            # second cumulative score: each client against its OWN first checked round
            zbase, scus = st.setdefault('zbase', {}), st.setdefault('scus', {})
            szero = st.setdefault('scus_zero', {})
            self_cand = []
            for k, v in sc.items():
                if k not in zbase:
                    zbase[k] = v['z']
                    szero[k] = rnd
                change = v['z'] - zbase[k]
                scus[k] = max(0.0, scus.get(k, 0.0) + change - args.self_drift)
                if scus[k] == 0.0:
                    szero[k] = rnd
                if scus[k] >= args.self_h and v['score'] >= args.t_floor:
                    self_cand.append((scus[k] + change, k))
            for k in range(K):
                row[f'zchange_c{k}'] = sc[k]['z'] - zbase[k] if k in sc else float('nan')
                row[f'selfcusum_c{k}'] = scus.get(k, float('nan')) if k in sc else float('nan')
            both = {k for _, k in cum_cand} | {k for _, k in self_cand}
            row['would_flag_both'] = ' '.join(str(k) for k in sorted(both))
            if both and method == 'detect_both':
                # strongest evidence first: the client with the largest sum of both scores
                flagged = max(both, key=lambda k: cus.get(k, 0.0) + scus.get(k, 0.0))

        # snapshot of the model at the end of this round (before any rollback decision)
        st['snaps'][rnd] = {'bottom': cpu_state(bottoms[0]), 'server': cpu_state(server),
                            's_opt': copy.deepcopy(s_opt.state_dict()),
                            'b_opts': [copy.deepcopy(o.state_dict()) for o in b_opts]}
        for old in [r for r in st['snaps'] if r < rnd - args.keep]:
            del st['snaps'][old]
        st['round'] = rnd

        if flagged is not None:
            first = rnd
            zh = st['zhist'].get(flagged, {})
            while first - 1 >= args.detect_from and zh.get(first - 1, float('nan')) >= args.z_soft:
                first -= 1
            back = first - 1
            if method in ('detect_cum', 'detect_both'):
                # also go back to the last round in which the client's cumulative score was still zero
                back = min(back, st.get('cusum_zero', {}).get(flagged, back))
            if method == 'detect_both':
                back = min(back, st.get('scus_zero', {}).get(flagged, back))
            back = max(min(st['snaps']), back)            # oldest snapshot we still hold bounds the rollback
            snap = st['snaps'][back]
            for b in bottoms:
                b.load_state_dict(snap['bottom'])
            server.load_state_dict(snap['server'])
            s_opt.load_state_dict(snap['s_opt'])
            b_opts = [optim.Adam(b.parameters(), lr=args.lr) for b in bottoms]
            if not args.reset_opt and 'b_opts' in snap:
                # give every client its optimiser state (Adam moments) from that round back as well;
                # starting them from zero after a rollback cost up to ~2 points of accuracy
                for o, sd in zip(b_opts, snap['b_opts']):
                    o.load_state_dict(copy.deepcopy(sd))
            st['excluded'].append(flagged)
            st['flags'].append({'client': flagged, 'round': rnd, 'z': float(sc[flagged]['z']), 'rollback_to': back})
            for r in [r for r in st['snaps'] if r > back]:
                del st['snaps'][r]
            st['round'] = back
            st['cusum'] = {}                               # rounds after `back` are trained again: restart the sums
            st['cusum_zero'] = {k: back for k in range(K)}
            st['zbase'], st['scus'], st['scus_zero'] = {}, {}, {}   # compared clients changed: new baselines
            row['flagged'], row['rollback_to'] = str(flagged), back

        st['log'].append(row)
        msg = f"    round {rnd}/{R} (step {st['step']})  loss {row['loss']:.4f}"
        if rnd >= args.detect_from and any(f'z_c{k}' in row for k in range(K)):
            zs_txt = ' '.join(f"c{k}:{row[f'z_c{k}']:.1f}" for k in range(K) if not np.isnan(row[f'z_c{k}']))
            msg += f" | z {zs_txt}"
        if row['poisoning']:
            msg += " | poison ON"
        if flagged is not None:
            msg += f" | FLAGGED client {flagged} -> excluded, rollback to round {row['rollback_to']}"
        print(msg, flush=True)
        if ckpt_path:
            atomic_save({'signature': signature, 'bottoms': [b.state_dict() for b in bottoms],
                         'b_opts': [o.state_dict() for o in b_opts], 'server': server.state_dict(),
                         's_opt': s_opt.state_dict(), 'rng': _rng_state(), 'st': st}, ckpt_path)

    acc, asr = clean_acc_and_asr(bottoms[0], server, raw['test_plain'], args.target, trigger, device,
                                 args.batch_size, mb)
    flags = st['flags']
    # the malicious client counts as correctly found only if it was flagged while it was poisoning
    def real(f_round):
        return n_poison > 0 and f_round >= args.poison_start
    mal_flag = next((f for f in flags if f['client'] == mal and real(f['round'])), None)
    would = [r for r in st['log'] if str(mal) in str(r.get('would_flag', '')).split() and real(r['round'])]
    res = {'dataset': dataset, 'model': model, 'cut': cut, 'poison_rate': rate, 'method': method,
           'target': args.target, 'clients': K, 'malicious': mal, 'poison_start': args.poison_start,
           'n_poison': int(n_poison), 'poison_share_of_shard': n_poison / max(len(shards[mal]), 1),
           'malicious_shard_size': int(len(shards[mal])),
           'data': 'IID' if args.noniid is None else f"non-IID Dirichlet {args.noniid:g}",
           'opt_restored': not args.reset_opt,
           'clean_acc': acc, 'asr': asr,
           'flagged': ' '.join(f"c{f['client']}@r{f['round']}" for f in flags),
           'malicious_flagged': bool(mal_flag) if method != 'none' else bool(would),
           'flag_round': mal_flag['round'] if mal_flag else (would[0]['round'] if would else float('nan')),
           'false_flags': sum(1 for f in flags if f['client'] != mal or not real(f['round'])),
           'rollback_to': mal_flag['rollback_to'] if mal_flag else float('nan'),
           'rounds_executed': st['step'], 'rounds': R, 'z_thr': args.z_thr,
           'cusum_drift': args.cusum_drift if method in ('detect_cum', 'detect_both') else float('nan'),
           'cusum_h': args.cusum_h if method in ('detect_cum', 'detect_both') else float('nan'),
           'self_drift': args.self_drift if method == 'detect_both' else float('nan'),
           'self_h': args.self_h if method == 'detect_both' else float('nan'), 'seed': args.seed,
           'seconds': round(time.time() - t0, 1)}
    res['detection_delay'] = res['flag_round'] - args.poison_start if n_poison and not np.isnan(res['flag_round']) \
        else float('nan')
    if method == 'none':    # monitor only: which benign clients WOULD have been flagged at some round
        benign = set()
        for r in st['log']:
            benign.update(k for k in str(r.get('would_flag', '')).split() if k != str(mal) or not real(r['round']))
        res['false_flags'] = len(benign)
    print(f"  -> clean acc {acc:.2f}% | ASR {asr:.2f}% | flagged [{res['flagged']}] | "
          f"malicious flagged: {res['malicious_flagged']} | false flags: {res['false_flags']}", flush=True)
    log = [dict(r, dataset=dataset, model=model, cut=cut, poison_rate=rate, method=method) for r in st['log']]
    del bottoms, server
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res, log, ckpt_path


def main():
    ap = argparse.ArgumentParser(description="Experiment 3: client-level backdoor detection in multi-client SL")
    ap.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10', 'MNIST'])
    ap.add_argument('--models', nargs='+', default=['KAGN', 'PyramidCNN'])
    ap.add_argument('--cuts', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--rates', type=float, nargs='+', default=[0.0, 0.01, 0.05, 0.10],
                    help="poison as a share of the WHOLE training set (all of it inside the malicious client)")
    ap.add_argument('--methods', nargs='+', default=list(METHODS), choices=METHODS)
    ap.add_argument('--clients', type=int, default=5)
    ap.add_argument('--malicious', type=int, default=2, help="index of the malicious client (0-based)")
    ap.add_argument('--poison-start', type=int, default=3, help="round in which the client starts poisoning")
    ap.add_argument('--rounds', type=int, default=10)
    ap.add_argument('--detect-from', type=int, default=2, help="first round whose end the detector checks")
    ap.add_argument('--z-thr', type=float, default=8.0, help="robust z-score at which a client is flagged")
    ap.add_argument('--z-soft', type=float, default=4.0,
                    help="rollback walks back over earlier rounds while the client's z stayed above this")
    ap.add_argument('--t-floor', type=float, default=2.0, help="minimum raw score (in standard errors) to flag")
    ap.add_argument('--cusum-drift', type=float, default=4.0,
                    help="detect_cum: z-score an honest client may show every round without building up a score")
    ap.add_argument('--cusum-h', type=float, default=4.0,
                    help="detect_cum: cumulative score at which a client is flagged (z >= --z-thr still flags at once)")
    ap.add_argument('--self-drift', type=float, default=1.5,
                    help="detect_both: rise of a client's z above its own first checked round that is still "
                         "allowed every round without building up the self score")
    ap.add_argument('--self-h', type=float, default=3.0,
                    help="detect_both: self score at which a client is flagged")
    ap.add_argument('--keep', type=int, default=4, help="how many past rounds' checkpoints are kept for rollback")
    ap.add_argument('--target', type=int, default=0)
    ap.add_argument('--trigger-size', type=int, default=4)
    ap.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    ap.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noniid', type=float, default=None,
                    help="non-IID label skew: Dirichlet concentration (e.g. 0.5; smaller = more skew). "
                         "Default: IID equal shards")
    ap.add_argument('--reset-opt', action='store_true',
                    help="old behaviour: restart the clients' optimisers from zero after a rollback")
    ap.add_argument('--tag', default=None,
                    help="suffix for the result files, to keep experiment variants apart. Default: built "
                         "from --noniid, --clients (if not 5) and --seed (if not 0)")
    ap.add_argument('--drop-method', default=None, choices=METHODS,
                    help="remove this method's rows from the result files (a .bak copy is kept) and exit, "
                         "so that the method can be run again")
    ap.add_argument('--keep-ckpt', action='store_true')
    ap.add_argument('--quick', action='store_true', help="smoke test: 4 short rounds, one config")
    args = ap.parse_args()

    args.max_batches = None
    if args.quick:
        args.rounds, args.poison_start, args.detect_from, args.max_batches = 4, 2, 1, 25
        args.models, args.cuts, args.rates = args.models[:1], args.cuts[:1], [0.05]
    if not 0 <= args.malicious < args.clients:
        sys.exit("--malicious must be between 0 and --clients - 1")

    os.chdir(ROOT)
    device = get_device()
    args.out_dir = os.path.join(Config.RESULTS_DIR, 'exp3_client')
    args.ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.tag is None:
        parts = ([f"noniid{args.noniid:g}"] if args.noniid is not None else []) + \
                ([f"k{args.clients}"] if args.clients != 5 else []) + \
                ([f"seed{args.seed}"] if args.seed != 0 else [])
        args.tag = '_'.join(parts)
    args.fsuf = f"_{args.tag}" if args.tag else ''
    suf = args.fsuf + ('_quick' if args.quick else '')
    out_csv = os.path.join(args.out_dir, f"exp3_results_{args.dataset}{suf}.csv")
    rd_csv = os.path.join(args.out_dir, f"exp3_rounds_{args.dataset}{suf}.csv")

    if args.drop_method:
        for path in (out_csv, rd_csv):
            if not os.path.exists(path):
                print(f"[drop] {path} not found, nothing to do")
                continue
            d = pd.read_csv(path)
            keep = d[d['method'] != args.drop_method]
            d.to_csv(path + '.bak', index=False)
            atomic_csv(keep.to_dict('records'), path)
            print(f"[drop] {path}: removed {len(d) - len(keep)} rows of method '{args.drop_method}' "
                  f"(backup: {path}.bak)")
        return

    raw = load_raw_datasets(args.dataset)
    rows = pd.read_csv(out_csv).to_dict('records') if os.path.exists(out_csv) and not args.quick else []
    rd_rows = pd.read_csv(rd_csv).to_dict('records') if os.path.exists(rd_csv) and not args.quick else []
    done = {(r['model'], int(r['cut']), float(r['poison_rate']), r['method']) for r in rows}
    todo = [(m, c, r, u) for m in args.models for c in args.cuts for r in args.rates for u in args.methods]
    print(f"[plan] {len(todo)} runs, {sum(t in done for t in todo)} already finished, results -> {out_csv}")

    try:
        for i, (model, cut, rate, method) in enumerate(todo, 1):
            if (model, cut, rate, method) in done:
                print(f"[skip {i}/{len(todo)}] {model} cut {cut} rate {rate:g} method {method} already finished")
                continue
            print(f"\n=== run {i}/{len(todo)} | {args.dataset} | {model} | cut {cut} | poison {rate:g} | "
                  f"method {method} ===", flush=True)
            res, log, ckpt = run_one(raw, args.dataset, model, cut, rate, method, args, device)
            rows.append(res)
            rd_rows.extend(log)
            atomic_csv(rd_rows, rd_csv)
            atomic_csv(rows, out_csv)         # the run counts as finished only once it is in the CSV
            if ckpt and os.path.exists(ckpt) and not args.keep_ckpt:
                os.remove(ckpt)
    except KeyboardInterrupt:
        print("\n[stopped] Run the same command again to continue: finished runs are skipped and the "
              "interrupted run resumes from its last finished round.")
        sys.exit(130)

    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print("\n" + df[['model', 'cut', 'poison_rate', 'method', 'clean_acc', 'asr', 'flagged', 'malicious_flagged',
                     'false_flags', 'detection_delay']].to_string(index=False))
    print(f"\nsaved -> {out_csv}")
    try:
        from exp3_report import build_report
        print(f"PDF report -> {build_report(out_csv, rd_csv)}")
    except Exception as exc:
        print(f"[warn] PDF report failed ({type(exc).__name__}: {exc}); "
              f"rebuild with: python experiments/exp3_report.py {out_csv}")


if __name__ == '__main__':
    main()