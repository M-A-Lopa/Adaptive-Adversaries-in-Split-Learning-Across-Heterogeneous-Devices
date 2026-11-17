"""
Single-client vs multi-client: the undefended attack results side by side.

Reads the single-client suite (defence_runner/main_final.py, default results/undefended_suite) and the
multi-client suite (multiclient_attacks_run/mc_suite.py, results/mc/<setting>/undefended_suite) and
writes, next to the multi-client results:
    compare_single_vs_multi.csv
    compare_single_vs_multi.pdf
For every attack and configuration: the attack metric (SSIM, leak AUC, label accuracy, ASR), its
no-information floor and whether the attack counts as working (util/threshold_rule.py), in both setups.

Run from the project root:
    python multiclient_attacks_run/mc_report.py
    python multiclient_attacks_run/mc_report.py --setting K5_noniid0.5
    python multiclient_attacks_run/mc_report.py --single-root results/undefended_suite
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, 'defence_runner')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import argparse                                        # noqa: E402
from datetime import datetime                          # noqa: E402

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages   # noqa: E402

import main_final as MF                                # noqa: E402

KEY = ['Dataset', 'Model', 'Cut', 'Attack']


def frame(root, label):
    rows = MF.collect(root, MF.DATASETS, MF.MODELS, MF.CUTS)
    if not rows:
        return None
    f = MF.floors_frame(rows)
    f = f[f['Undefended'].notna()]
    return f[KEY + ['Metric', 'Floor', 'Undefended', 'Valid']].rename(
        columns={'Floor': f'Floor {label}', 'Undefended': label, 'Valid': f'Works {label}'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='K5_iid', help='K5_iid, K5_noniid0.5, ...')
    ap.add_argument('--single-root', default=os.path.join('results', 'undefended_suite'))
    ap.add_argument('--quick', action='store_true', help='read the --quick test run')
    a = ap.parse_args()
    multi_root = os.path.join('results', 'mc', a.setting, 'undefended_suite' + ('_quick' if a.quick else ''))
    single = frame(a.single_root, 'single')
    multi = frame(multi_root, 'multi')
    if multi is None:
        sys.exit(f"no multi-client results in {multi_root}")
    if single is None:
        print(f"[warn] no single-client results in {a.single_root}; showing multi-client only "
              f"(use --single-root to point at them)")
        df = multi
    else:
        df = multi.merge(single.drop(columns=['Metric']), on=KEY, how='left')
        df['Change'] = df['multi'] - df['single']

        def verdict(r):
            s, m = r.get('Works single'), r['Works multi']
            if s not in ('yes', 'NO'):
                return 'no single-client result'
            if s == m:
                return 'same'
            return 'STOPS WORKING' if s == 'yes' else 'STARTS WORKING'
        df['Verdict'] = df.apply(verdict, axis=1)
        cols = KEY + ['Metric', 'single', 'multi', 'Change', 'Floor single', 'Floor multi',
                      'Works single', 'Works multi', 'Verdict']
        df = df[cols]
    df = df.sort_values(KEY).reset_index(drop=True)
    out_csv = os.path.join(multi_root, 'compare_single_vs_multi.csv')
    df.to_csv(out_csv, index=False)

    out_pdf = os.path.join(multi_root, 'compare_single_vs_multi.pdf')
    with PdfPages(out_pdf) as pdf:
        lines = [f"Generated {datetime.now():%Y-%m-%d %H:%M:%S}",
                 f"Single-client results : {a.single_root}",
                 f"Multi-client results  : {multi_root}", '']
        if 'Verdict' in df:
            lines.append('Per attack (configurations where both setups have a result):')
            for att, g in df[df['Verdict'] != 'no single-client result'].groupby('Attack'):
                n = len(g)
                ws, wm = int((g['Works single'] == 'yes').sum()), int((g['Works multi'] == 'yes').sum())
                lines.append(f"  {att:<40} works single {ws}/{n}  ->  multi {wm}/{n} | "
                             f"mean metric {g['single'].mean():.3f} -> {g['multi'].mean():.3f}")
            lines += ['', 'Metric: SSIM (reconstruction), leak AUC (label leakage), cluster accuracy (similarity label),',
                      'ASR % (backdoor). Works = threshold rule of util/threshold_rule.py (metric clearly above its floor).',
                      'STOPS WORKING / STARTS WORKING = the verdict differs between the two setups.']
        MF.text_page(pdf, 'Single-client vs multi-client (undefended)', lines)
        show = df.copy()
        for c in show.columns:
            if show[c].dtype.kind == 'f':
                show[c] = show[c].map(lambda v: MF.fmt(v, '.3f'))
        MF.table_pages(pdf, 'Undefended attacks: single-client vs multi-client', show, fontsize=6, rows_per_page=28)
    print(f"saved -> {out_csv}\nsaved -> {out_pdf}")


if __name__ == '__main__':
    main()