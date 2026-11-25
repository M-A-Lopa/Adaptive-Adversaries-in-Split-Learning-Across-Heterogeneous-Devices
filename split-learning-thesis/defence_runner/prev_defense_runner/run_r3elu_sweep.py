import os
import sys
import time
import argparse
import traceback


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
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, ABLATION_DIR)

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
import torch

import abl_common as C
from config import Config
from abl_report import build_pdf, load_sweep
from abl_r3elu import R3eLUAblation

MODELS = ['KAGN', 'PyramidCNN']
ATTACKS = ['whitebox', 'ae_decoder', 'fsha', 'label_leakage']
KEY = ('model', 'variant', 'attack')


def merge(previous, new):
    replaced = {tuple(r[k] for k in KEY) for r in new}
    return [r for r in previous if tuple(r.get(k) for k in KEY) not in replaced] + list(new)


def run_point(abl, variant, attack, ctx):
    t0 = time.time()
    try:
        C.seed_all(ctx['seed'])
        r = abl.run(variant, attack, ctx)
        if attack == 'label_leakage':
            r['task_acc'] = r.get('acc')
            r['acc'] = abl._trained(variant, ctx)[2]
        r['status'] = 'error' if 'error' in r else 'ok'
    except Exception as exc:
        traceback.print_exc()
        r = {'status': 'error', 'error': f"{type(exc).__name__}: {exc}"}
    r['seconds'] = round(time.time() - t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='both', help="KAGN, PyramidCNN or both")
    ap.add_argument('--eps', default='1,10,100,1000,10000,100000', help="comma list of epsilon values")
    ap.add_argument('--attacks', default=','.join(ATTACKS))
    ap.add_argument('--sensitivity', default='global', choices=['global', 'per_coord'])
    ap.add_argument('--usable-tol', type=float, default=5.0, help="accuracy points below no-defense still called usable")
    ap.add_argument('--cut', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--def-epochs', type=int, default=C.SETTINGS['def_epochs'])
    ap.add_argument('--ae-epochs', type=int, default=C.SETTINGS['ae_epochs'])
    ap.add_argument('--fsha-epochs', type=int, default=C.SETTINGS['fsha_epochs'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--report-only', action='store_true')
    args = ap.parse_args()

    os.chdir(C.ROOT)
    eps_values = [float(e) for e in args.eps.split(',') if e.strip()]
    attacks = [a for a in args.attacks.split(',') if a]
    models = MODELS if args.model == 'both' else [args.model]
    C.SETTINGS.update(def_epochs=args.def_epochs, ae_epochs=args.ae_epochs, fsha_epochs=args.fsha_epochs)
    if args.quick:
        C.SETTINGS.update(def_epochs=1, attack_epochs=1, ae_epochs=2, fsha_epochs=1, wb_iters=20, max_images=4,
                          leak_epochs=1, vanilla_epochs=1, max_batches=5)

    suffix = '_quick' if args.quick else ''
    out_dir = os.path.join(Config.RESULTS_DIR, 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    sweep_csv = os.path.join(out_dir, f"r3elu_eps_sweep{suffix}.csv")
    summary_csv = os.path.join(out_dir, f"ablation_summary{suffix}.csv")
    pdf_path = os.path.join(out_dir, f"ablation_pdf{suffix}.pdf")

    def rebuild(sweep_rows):
        main_rows = pd.read_csv(summary_csv).to_dict('records') if os.path.exists(summary_csv) else []
        meta = {'cut': args.cut, 'seed': args.seed, 'gates': C.GATES, 'usable_tol': args.usable_tol,
                'budgets': ', '.join(f"{k}={v}" for k, v in C.SETTINGS.items()), 'sweep_rows': sweep_rows}
        try:
            print(f"[sweep] PDF -> {build_pdf(main_rows, pdf_path, meta)}")
        except Exception:
            traceback.print_exc()

    previous = load_sweep(sweep_csv)
    if args.report_only:
        rebuild(previous)
        return

    new_rows = []
    for model in models:
        device = C.setup(model, 'CIFAR10', args.cut, args.seed)
        ctx = {'device': device, 'seed': args.seed, 'cache': {}}
        ctx['train'], ctx['test'] = C.loaders()
        points = [('off', None)] + [(f"eps_{e:g}", e) for e in eps_values]
        for variant, eps in points:
            abl = R3eLUAblation(epsilon=eps if eps is not None else 1.0, sensitivity=args.sensitivity)
            run_variant = 'off' if eps is None else 'full'
            for attack in attacks:
                print(f"\n{'#' * 80}\n# R3eLU sweep | {model} | {variant} | {attack}\n{'#' * 80}")
                r = run_point(abl, run_variant, attack, ctx)
                row = {'model': model, 'dataset': 'CIFAR10', 'cut_layer': args.cut, 'seed': args.seed,
                       'variant': variant, 'epsilon': eps, 'sensitivity': args.sensitivity, 'attack': attack,
                       'metric': C.METRIC_OF[attack], 'value': r.get(C.METRIC_OF[attack])}
                row.update({k: v for k, v in r.items() if k not in row})
                new_rows.append(row)
                combined = merge(previous, new_rows)
                pd.DataFrame(combined).to_csv(sweep_csv, index=False)
        del ctx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        rebuild(merge(previous, new_rows))

    df = pd.DataFrame(new_rows)
    if not df.empty:
        print("\n" + df.pivot_table(index=['model', 'variant'], columns='attack', values='value', aggfunc='first').to_string())
        print("\n" + df.groupby(['model', 'variant'])['acc'].first().to_string())
    print(f"\n[sweep] results: {sweep_csv}\n[sweep] PDF: {pdf_path}")


if __name__ == '__main__':
    main()