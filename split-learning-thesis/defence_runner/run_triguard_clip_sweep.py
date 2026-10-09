import os
import sys
import argparse
import subprocess


def _find_dir(marker_file, subdirs):
    d = os.path.dirname(os.path.abspath(__file__))
    while True:
        for sub in subdirs:
            cand = os.path.join(d, sub)
            if os.path.isfile(os.path.join(cand, marker_file)):
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError(f"could not find {marker_file}")
        d = parent


PROJECT_ROOT = _find_dir('config.py', [''])
ABLATION_DIR = _find_dir('abl_common.py', ['ablation', os.path.join('defence_runner', 'ablation')])
for _p in (PROJECT_ROOT, ABLATION_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import abl_common as C
from config import Config

MODELS = ['KAGN', 'PyramidCNN']
LEAK_FLOOR = 0.69
LEAK_TOL = 0.05
ACC_TOL = 5.0


def probe(model, epsilons, clips, batches=20):
    import torch
    import torch.nn as nn
    device = C.setup(model, 'CIFAR10', 2, 0)
    tr, _ = C.loaders()
    client, server, _ = C.load_vanilla(device)
    client.train()
    server.train()
    crit = nn.CrossEntropyLoss()
    vals, sample_max = [], []
    for i, (x, y) in enumerate(tr):
        if i >= batches:
            break
        x, y = x.to(device), y.to(device)
        z = client(x)
        z_leaf = z.detach().requires_grad_(True)
        crit(server(z_leaf), y).backward()
        g = z_leaf.grad.detach().reshape(z_leaf.shape[0], -1).abs()
        vals.append(g.flatten().cpu())
        sample_max.append(g.max(dim=1).values.cpu())
        client.zero_grad(set_to_none=True)
        server.zero_grad(set_to_none=True)
    v = torch.cat(vals).numpy()
    m = torch.cat(sample_max).numpy()
    nf = C.n_features(client, device)
    K = max(1, nf // 2)
    row = {'model': model, 'num_features': nf, 'K': K,
           'grad_abs_median': float(np.median(v)), 'grad_abs_p90': float(np.quantile(v, 0.90)),
           'grad_abs_p99': float(np.quantile(v, 0.99)), 'grad_abs_p999': float(np.quantile(v, 0.999)),
           'grad_abs_max': float(v.max()), 'per_sample_max_median': float(np.median(m))}
    for c in clips:
        for e in epsilons:
            scale = 2.0 * K * c / (e / 2.0)
            row[f'noise_scale_C{c:g}_eps{e:g}'] = scale
            row[f'noise_to_signal_C{c:g}_eps{e:g}'] = scale / max(row['grad_abs_median'], 1e-12)
    return row


def run_one(clip, eps, seed, args, subdir):
    script = os.path.join(PROJECT_ROOT, 'defence_runner', 'run_triguard.py')
    cmd = [sys.executable, script, '--clip-c', f'{clip:g}', '--seed', str(seed), '--epsilon', f'{eps:g}',
           '--attacks', args.attacks, '--out-subdir', subdir, '--fresh', '--model', args.model]
    if args.quick:
        cmd.append('--quick')
    print('[clip-sweep] ' + ' '.join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def collect(args, clips, epsilons, seeds, suffix):
    rows = []
    for c in clips:
        for e in epsilons:
            for s in seeds:
                path = os.path.join(Config.RESULTS_DIR, 'triguard', subdir_name(c, e, s), f"triguard_summary{suffix}.csv")
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                for _, r in df.iterrows():
                    rows.append({'clip_c': c, 'epsilon': e, 'seed': s, 'model': r['model'], 'attack': r['attack'],
                                 'variant': r['variant'], 'value': r.get('value'), 'acc': r.get('acc'),
                                 'responsibility': r.get('responsibility'), 'status': r.get('status')})
    return pd.DataFrame(rows)


def subdir_name(c, e, s):
    return os.path.join('clip_sweep', f"C{c:g}_eps{e:g}_seed{s}")


def summarise(df, K=2048):
    out = []
    for (m, c, e), g in df.groupby(['model', 'clip_c', 'epsilon']):
        def pick(attack, variant, col):
            x = g[(g['attack'] == attack) & (g['variant'] == variant)][col]
            x = pd.to_numeric(x, errors='coerce').dropna()
            return (float(x.mean()), float(x.std()) if len(x) > 1 else 0.0, len(x)) if len(x) else (np.nan, np.nan, 0)
        leak, leak_sd, n = pick('label_leakage', 'full', 'value')
        asr, asr_sd, _ = pick('poison_client', 'full', 'value')
        asr_off, _, _ = pick('poison_client', 'off', 'value')
        acc, acc_sd, _ = pick('poison_client', 'full', 'acc')
        acc_off, _, _ = pick('poison_client', 'off', 'acc')
        r_noise, _, _ = pick('poison_client', 'no_backward_laplace', 'responsibility')
        r_heads, _, _ = pick('poison_client', 'shared_classifier_heads', 'responsibility')
        out.append({'model': m, 'clip_c': c, 'epsilon': e, 'noise_scale': 4.0 * K * c / e, 'seeds': n, 'leak_auc': leak, 'leak_auc_sd': leak_sd,
                    'asr': asr, 'asr_sd': asr_sd, 'asr_off': asr_off, 'acc': acc, 'acc_sd': acc_sd,
                    'acc_off': acc_off, 'resp_backward_noise': r_noise, 'resp_private_heads': r_heads})
    return pd.DataFrame(out).sort_values(['model', 'noise_scale'], ascending=[True, False])


def recommend(summary):
    rec = {}
    for m, g in summary.groupby('model'):
        ok = g[(g['leak_auc'] <= LEAK_FLOOR + LEAK_TOL)]
        if ok.empty:
            rec[m] = None
            continue
        best = ok.sort_values(['acc', 'asr'], ascending=[False, True]).iloc[0]
        rec[m] = best
    return rec


def _fmt(v, nd=3):
    return '-' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def build_pdf(path, probe_rows, summary, rec, args):
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.04, 0.95, 'TriGuard-SL: clipping-bound (C) sweep for the backward Laplace noise', fontsize=15, weight='bold')
        text = (f"Grid: C in {args.clips}, epsilon in {args.epsilons}. Noise scale = 2*K*C/(epsilon/2). C is set near the measured "
                f"99th percentile of the cut-layer gradients; epsilon then sets how much noise is added relative to them.\n"
                f"Selection rule: keep label-leak AUC <= {LEAK_FLOOR + LEAK_TOL:.2f} (random-guessing level of the 95%-quantile "
                f"metric is about {LEAK_FLOOR:.2f}), then take the highest accuracy, then the lowest backdoor ASR.")
        fig.text(0.04, 0.88, text, fontsize=8.5, wrap=True)
        y = 0.80
        if probe_rows:
            fig.text(0.04, y, 'Measured cut-layer gradient sizes (vanilla model)', fontsize=11, weight='bold')
            cols = ['model', 'grad_abs_median', 'grad_abs_p90', 'grad_abs_p99', 'grad_abs_max', 'K']
            cells = [[r['model']] + [f"{r[k]:.3g}" for k in cols[1:]] for r in probe_rows]
            ax = fig.add_axes([0.04, y - 0.14, 0.92, 0.12])
            ax.axis('off')
            tb = ax.table(cellText=cells, colLabels=cols, loc='upper left', cellLoc='center')
            tb.auto_set_font_size(False)
            tb.set_fontsize(7)
            y -= 0.20
        fig.text(0.04, y, 'Recommended C per model', fontsize=11, weight='bold')
        y -= 0.04
        for m, r in rec.items():
            if r is None:
                line = f"{m}: no tested C keeps label-leak AUC at the random-guessing level."
            else:
                line = (f"{m}: C = {r['clip_c']:g}, epsilon = {r['epsilon']:g} (noise scale {r['noise_scale']:.2g}) | leak AUC {_fmt(r['leak_auc'])} | accuracy {_fmt(r['acc'], 2)}% "
                        f"(no defense {_fmt(r['acc_off'], 2)}%) | backdoor ASR {_fmt(r['asr'], 2)}% "
                        f"(no defense {_fmt(r['asr_off'], 2)}%) | responsibility: backward noise "
                        f"{_fmt(r['resp_backward_noise'], 2)}, private heads {_fmt(r['resp_private_heads'], 2)}")
            fig.text(0.04, y, line, fontsize=8.5)
            y -= 0.035
        pdf.savefig(fig)
        plt.close(fig)
        for m, g in summary.groupby('model'):
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.text(0.04, 0.95, f"{m}: results per noise setting (mean over seeds)", fontsize=13, weight='bold')
            cols = ['C', 'epsilon', 'noise scale', 'seeds', 'leak AUC', 'acc %', 'acc % no defense', 'ASR %', 'ASR % no defense',
                    'resp. backward noise', 'resp. private heads']
            cells = [[f"{r['clip_c']:g}", f"{r['epsilon']:g}", f"{r['noise_scale']:.2g}", int(r['seeds']), f"{_fmt(r['leak_auc'])} ± {_fmt(r['leak_auc_sd'])}",
                      f"{_fmt(r['acc'], 2)} ± {_fmt(r['acc_sd'], 2)}", _fmt(r['acc_off'], 2),
                      f"{_fmt(r['asr'], 2)} ± {_fmt(r['asr_sd'], 2)}", _fmt(r['asr_off'], 2),
                      _fmt(r['resp_backward_noise'], 2), _fmt(r['resp_private_heads'], 2)] for _, r in g.iterrows()]
            ax = fig.add_axes([0.04, 0.62, 0.92, 0.28])
            ax.axis('off')
            tb = ax.table(cellText=cells, colLabels=cols, loc='upper left', cellLoc='center')
            tb.auto_set_font_size(False)
            tb.set_fontsize(7)
            tb.scale(1, 1.4)
            gs = g.sort_values('noise_scale')
            for i, (key, lab, ref) in enumerate([('acc', 'accuracy %', 'acc_off'), ('asr', 'backdoor ASR %', 'asr_off'),
                                                 ('leak_auc', 'label-leak AUC', None)]):
                ax = fig.add_axes([0.06 + i * 0.32, 0.10, 0.26, 0.42])
                ax.plot(gs['noise_scale'], gs[key], marker='o', label='TriGuard-SL')
                if ref:
                    ax.plot(gs['noise_scale'], gs[ref], ls='--', color='grey', label='no defense')
                else:
                    ax.axhline(LEAK_FLOOR, ls=':', color='green', label='random guessing')
                    ax.axhline(1.0, ls='--', color='grey', label='no defense')
                ax.set_xscale('log')
                ax.set_xlabel('noise scale (log scale)')
                ax.set_title(lab, fontsize=10)
                ax.legend(fontsize=7)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', default='0.001')
    ap.add_argument('--epsilons', default='1000,10000,100000')
    ap.add_argument('--seeds', default='0,1')
    ap.add_argument('--attacks', default='label_leakage,poison_client')
    ap.add_argument('--model', default='both')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--probe-only', action='store_true')
    ap.add_argument('--skip-probe', action='store_true')
    ap.add_argument('--rerun', action='store_true', help="rerun settings that already have results")
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()

    os.chdir(C.ROOT)
    clips = [float(c) for c in args.clips.split(',') if c.strip()]
    epsilons = [float(e) for e in args.epsilons.split(',') if e.strip()]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    models = MODELS if args.model == 'both' else [args.model]
    suffix = '_quick' if args.quick else ''
    out_dir = os.path.join(Config.RESULTS_DIR, 'triguard', 'clip_sweep')
    os.makedirs(out_dir, exist_ok=True)
    probe_csv = os.path.join(out_dir, 'gradient_probe.csv')

    probe_rows = []
    if not args.skip_probe and not args.report_only:
        for m in models:
            r = probe(m, epsilons, clips)
            probe_rows.append(r)
            print(f"\n[probe] {m}: |grad| median {r['grad_abs_median']:.3g}, p99 {r['grad_abs_p99']:.3g}, "
                  f"max {r['grad_abs_max']:.3g}, K = {r['K']}")
            for c in clips:
                for e in epsilons:
                    print(f"  C = {c:g}, epsilon = {e:g}: noise scale {r[f'noise_scale_C{c:g}_eps{e:g}']:.3g}, "
                          f"noise / median gradient = {r[f'noise_to_signal_C{c:g}_eps{e:g}']:.3g}")
            print(f"  suggestion: C around the 99th percentile = {r['grad_abs_p99']:.3g}")
        pd.DataFrame(probe_rows).to_csv(probe_csv, index=False)
    elif os.path.exists(probe_csv):
        probe_rows = pd.read_csv(probe_csv).to_dict('records')
    if args.probe_only:
        return

    if not args.report_only:
        for c in clips:
            for e in epsilons:
                for s in seeds:
                    done = os.path.join(Config.RESULTS_DIR, 'triguard', subdir_name(c, e, s), f"triguard_summary{suffix}.csv")
                    if os.path.exists(done) and not args.rerun:
                        print(f"[clip-sweep] C = {c:g}, epsilon = {e:g}, seed {s}: already done, skipping")
                        continue
                    code = run_one(c, e, s, args, subdir_name(c, e, s))
                    if code != 0:
                        print(f"[clip-sweep] C = {c:g}, epsilon = {e:g}, seed {s}: run_triguard.py exited with code {code}")

    df = collect(args, clips, epsilons, seeds, suffix)
    if df.empty:
        print("[clip-sweep] no results found")
        return
    df.to_csv(os.path.join(out_dir, f"clip_sweep_results{suffix}.csv"), index=False)
    K = int(probe_rows[0]['K']) if probe_rows else 2048
    summary = summarise(df, K)
    summary.to_csv(os.path.join(out_dir, f"clip_sweep_summary{suffix}.csv"), index=False)
    rec = recommend(summary)
    pdf = build_pdf(os.path.join(out_dir, f"clip_sweep_pdf{suffix}.pdf"), probe_rows, summary, rec, args)
    print("\n" + summary.to_string(index=False))
    for m, r in rec.items():
        print(f"[clip-sweep] recommended for {m}: " + ('none' if r is None else f"C = {r['clip_c']:g}, epsilon = {r['epsilon']:g}"))
    print(f"[clip-sweep] PDF -> {pdf}")


if __name__ == '__main__':
    main()