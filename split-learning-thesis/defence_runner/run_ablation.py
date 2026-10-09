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

DEFENSES = ['splitfss', 'zorro', 'splitml', 'r3elu', 'protoguard', 'protoguard_paper', 'securesplit']
MODELS = ['KAGN', 'PyramidCNN']
DATASETS = ['CIFAR10']


def build_defense(name, args):
    if name == 'r3elu':
        from abl_r3elu import R3eLUAblation
        return R3eLUAblation(epsilon=args.r3elu_eps, sensitivity=args.r3elu_sensitivity)
    if name == 'splitfss':
        from abl_splitfss import SplitFSSAblation
        return SplitFSSAblation()
    if name == 'splitml':
        from abl_splitml import SplitMLAblation
        return SplitMLAblation(rounds=args.splitml_rounds)
    if name == 'protoguard':
        from abl_protoguard import ProtoGuardAblation
        return ProtoGuardAblation(alpha=args.protoguard_alpha)
    if name == 'securesplit':
        from abl_securesplit import SecureSplitAblation
        return SecureSplitAblation(reducer=args.securesplit_reducer)
    if name == 'protoguard_paper':
        from abl_protoguard_paper import ProtoGuardPaperAblation
        return ProtoGuardPaperAblation(alpha=args.protoguard_alpha)
    if name == 'zorro':
        from abl_zorro import ZorroAblation
        return ZorroAblation()
    raise ValueError(name)


def variants_for(defense, attack):
    if hasattr(defense, 'variants_for'):
        return defense.variants_for(attack)
    return list(defense.variants)


def verdict(resp):
    if resp is None:
        return '-'
    if resp >= 0.5:
        return 'RESPONSIBLE'
    if resp >= 0.2:
        return 'partial'
    if resp <= -0.2:
        return 'removal helps'
    return 'not responsible'


def run_one(defense, variant, attack, ctx):
    t0 = time.time()
    try:
        C.seed_all(ctx['seed'])
        r = defense.run(variant, attack, ctx)
        status = 'error' if 'error' in r else 'ok'
    except Exception as exc:
        traceback.print_exc()
        r, status = {'error': f"{type(exc).__name__}: {exc}"}, 'error'
    r['status'] = status
    r['seconds'] = round(time.time() - t0, 1)
    return r


def ablate(defense, args, ctx, out_rows, out_path):
    attacks = [a for a in defense.attacks if not args.attacks or a in args.attacks]
    for attack in attacks:
        metric = C.METRIC_OF[attack]
        names = variants_for(defense, attack)
        if args.variants:
            names = [v for v in names if v in args.variants or v in ('full', 'off')]
        res = {}
        for v in ('full', 'off'):
            print(f"\n{'#' * 80}\n# {defense.name} | {Config.MODEL_NAME} | {Config.DATASET} | {attack} | {v}\n{'#' * 80}")
            res[v] = run_one(defense, v, attack, ctx)
        ok = res['full']['status'] == 'ok' and res['off']['status'] == 'ok'
        gate = ok and C.substantial(attack, res['full'], res['off'])
        gap = C.protection(attack, res['full'], res['off']) if ok else None
        print(f"\n[gate] {defense.name} vs {attack}: protection ({metric}) = {gap} -> "
              f"{'substantial' if gate else 'NOT substantial'}{' (forced)' if args.force and not gate else ''}")
        for v in names:
            if v not in res:
                if gate or args.force:
                    print(f"\n{'#' * 80}\n# {defense.name} | {Config.MODEL_NAME} | {Config.DATASET} | {attack} | {v}\n{'#' * 80}")
                    res[v] = run_one(defense, v, attack, ctx)
                else:
                    res[v] = {'status': 'skipped (no substantial change for full defense)' if ok
                              else 'skipped (full/off run failed)'}
            r = res[v]
            resp = C.responsibility(attack, res['full'], res['off'], r) if r.get('status') == 'ok' and v not in ('full', 'off') else None
            row = {'defense': defense.name, 'model': Config.MODEL_NAME, 'dataset': Config.DATASET,
                   'cut_layer': Config.CUT_LAYER, 'seed': ctx['seed'], 'attack': attack, 'metric': metric,
                   'variant': v, 'value': r.get(metric), 'full_value': res['full'].get(metric),
                   'off_value': res['off'].get(metric), 'protection_gap': gap, 'gate_substantial': gate,
                   'responsibility': resp, 'verdict': 'reference' if v in ('full', 'off') else verdict(resp)}
            row.update({k: val for k, val in r.items() if k not in row})
            out_rows.append(row)
            pd.DataFrame(out_rows).to_csv(out_path, index=False)


def print_summary(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    print("\n" + "=" * 118)
    print(f"{'defense':<12}{'model':<11}{'data':<9}{'attack':<15}{'variant':<34}{'metric':<7}"
          f"{'value':>9}{'acc':>8}{'resp':>8}  verdict")
    print("-" * 118)
    for _, r in df.iterrows():
        val = '' if pd.isna(r.get('value')) else f"{r['value']:.4f}" if r['metric'] != 'asr' else f"{r['value']:.2f}"
        acc = '' if 'acc' not in r or pd.isna(r.get('acc')) else f"{r['acc']:.2f}"
        resp = '' if pd.isna(r.get('responsibility')) else f"{r['responsibility']:.2f}"
        v = r['verdict'] if r['status'] == 'ok' else r['status'][:28]
        print(f"{r['defense']:<12}{r['model']:<11}{r['dataset']:<9}{r['attack']:<15}{r['variant']:<34}"
              f"{r['metric']:<7}{val:>9}{acc:>8}{resp:>8}  {v}")
    print("=" * 118)
    ok = df[(df['verdict'] == 'RESPONSIBLE')]
    if not ok.empty:
        print("\nComponent -> attack responsibility (removing the component loses >= 50% of the protection):")
        for (d, m, ds, a), g in ok.groupby(['defense', 'model', 'dataset', 'attack']):
            print(f"  {d:<12}{m:<11}{ds:<9}{a:<15} " + ", ".join(g['variant']))


KEY = ('defense', 'model', 'dataset', 'attack')


def merge_rows(previous, new):
    replaced = {tuple(r[k] for k in KEY) for r in new}
    return [r for r in previous if tuple(r.get(k) for k in KEY) not in replaced] + list(new)


def load_previous(path, fresh):
    if fresh or not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict('records')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--defense', default='all', help="comma list of " + ','.join(DEFENSES) + " or all")
    ap.add_argument('--model', default='both', help="KAGN, PyramidCNN or both")
    ap.add_argument('--cut', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--attacks', default='', help="comma list to restrict attacks")
    ap.add_argument('--variants', default='', help="comma list to restrict variants (full/off always run)")
    ap.add_argument('--force', action='store_true', help="run component variants even without substantial change")
    ap.add_argument('--reuse', action='store_true', help="zorro: reuse existing run_zorro.py CSVs")
    ap.add_argument('--quick', action='store_true', help="smoke test with tiny budgets")
    ap.add_argument('--report-only', action='store_true', help="rebuild ablation_pdf.pdf from the saved summary CSV")
    ap.add_argument('--fresh', action='store_true', help="ignore earlier results instead of merging into them")
    ap.add_argument('--def-epochs', type=int, default=C.SETTINGS['def_epochs'])
    ap.add_argument('--attack-epochs', type=int, default=C.SETTINGS['attack_epochs'])
    ap.add_argument('--vanilla-epochs', type=int, default=C.SETTINGS['vanilla_epochs'])
    ap.add_argument('--ae-epochs', type=int, default=C.SETTINGS['ae_epochs'])
    ap.add_argument('--fsha-epochs', type=int, default=C.SETTINGS['fsha_epochs'])
    ap.add_argument('--r3elu-eps', type=float, default=1.0)
    ap.add_argument('--r3elu-sensitivity', default='global', choices=['global', 'per_coord'])
    ap.add_argument('--splitml-rounds', type=int, default=10)
    ap.add_argument('--protoguard-alpha', type=float, default=0.5)
    ap.add_argument('--securesplit-reducer', default='umap', choices=['umap', 'pca'])
    args = ap.parse_args()

    os.chdir(C.ROOT)
    args.attacks = [a for a in args.attacks.split(',') if a]
    args.variants = [v for v in args.variants.split(',') if v]
    defenses = DEFENSES if args.defense == 'all' else [d.strip() for d in args.defense.split(',')]
    models = MODELS if args.model == 'both' else [args.model]

    C.SETTINGS.update(def_epochs=args.def_epochs, attack_epochs=args.attack_epochs,
                      vanilla_epochs=args.vanilla_epochs, ae_epochs=args.ae_epochs, fsha_epochs=args.fsha_epochs)
    if args.quick:
        C.SETTINGS.update(def_epochs=1, attack_epochs=1, ae_epochs=2, fsha_epochs=1, wb_iters=20, max_images=4,
                          leak_epochs=1, vanilla_epochs=1, max_batches=5)
        args.splitml_rounds = 1
        if args.defense == 'all':
            defenses = [d for d in defenses if d != 'zorro']

    suffix = '_quick' if args.quick else ''
    out_dir = os.path.join(Config.RESULTS_DIR, 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, f"ablation_summary{suffix}.csv")
    pdf_path = os.path.join(out_dir, f"ablation_pdf{suffix}.pdf")
    meta = {'cut': args.cut, 'seed': args.seed, 'gates': C.GATES,
            'budgets': ', '.join(f"{k}={v}" for k, v in C.SETTINGS.items()) +
                       f", splitml_rounds={args.splitml_rounds}, r3elu_eps={args.r3elu_eps}, "
                       f"r3elu_sensitivity={args.r3elu_sensitivity}, protoguard_alpha={args.protoguard_alpha}, "
                       f"securesplit_reducer={args.securesplit_reducer}"}
    meta['sweep_rows'] = load_sweep(os.path.join(out_dir, f"r3elu_eps_sweep{suffix}.csv"))
    meta['usable_tol'] = 5.0

    if args.report_only:
        rows = pd.read_csv(summary_csv).to_dict('records')
        print(f"[ablation] PDF -> {build_pdf(rows, pdf_path, meta)}")
        print_summary(rows)
        return

    previous = load_previous(summary_csv, args.fresh)
    if previous:
        print(f"[ablation] merging into {len(previous)} earlier rows from {summary_csv}")
    all_rows = []
    combined = previous
    for dataset in DATASETS:
        for name in defenses:
            for model in models:
                device = C.setup(model, dataset, args.cut, args.seed)
                ctx = {'device': device, 'seed': args.seed, 'cache': {}, 'reuse': args.reuse}
                ctx['train'], ctx['test'] = C.loaders()
                defense = build_defense(name, args)
                rows = []
                path = os.path.join(out_dir, f"ablation_{name}_{model.lower()}_{dataset}_cut{args.cut}{suffix}.csv")
                try:
                    ablate(defense, args, ctx, rows, path)
                except Exception:
                    traceback.print_exc()
                all_rows += rows
                combined = merge_rows(previous, all_rows)
                pd.DataFrame(combined).to_csv(summary_csv, index=False)
                print(f"\n[ablation] saved -> {path}")
                del ctx
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            try:
                print(f"[ablation] PDF -> {build_pdf(combined, pdf_path, meta)}")
            except Exception:
                traceback.print_exc()
    print_summary(all_rows)
    print(f"\n[ablation] final PDF: {pdf_path}")


if __name__ == '__main__':
    main()