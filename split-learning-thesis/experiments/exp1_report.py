"""
PDF report for Experiment 1 (detector feasibility).

Called automatically at the end of exp1_detector_feasibility.py.  To rebuild the
report from saved results without re-running anything:

    python experiments/exp1_report.py results/exp1_detector/exp1_results_CIFAR10.csv

Pages
  1. verdict + full results table
  2. detector quality vs poison rate (separability AUC, spectral AUC) and backdoor check (ASR, clean accuracy)
  3. per run: target-class features on the top-2 principal directions, clean vs poisoned
  4. per run: histogram of the top-direction projection, clean vs poisoned
"""
import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ASR_OK = 80.0        # rows below this did not learn the backdoor -> nothing to detect
WORKS = 0.90         # separability AUC at or above this: features separate
WEAK = 0.70          # below this: features do not separate
CLEAN = '#4C78A8'
POISON = '#E45756'
LINE_COLORS = ['#4C78A8', '#F58518', '#54A24B', '#B279A2', '#9D755D', '#E45756']
A4 = (11.69, 8.27)


def _verdict(df):
    valid = df[df['asr'] >= ASR_OK]
    skipped = len(df) - len(valid)
    if valid.empty:
        return ('NO VALID RUNS', '#9D755D',
                f"No run reached ASR >= {ASR_OK:.0f}%, so the backdoor was never learned and there is nothing "
                f"to detect. Increase --epochs (e.g. 20) or check the trigger, then rerun.")
    sep = valid['sep_auc']
    n_good, n = int((sep >= WORKS).sum()), len(valid)
    if n_good == n:
        head, color = 'IDEA WORKS', '#54A24B'
        text = (f"In all {n} valid runs poisoned and clean target-class features separate "
                f"(separability AUC >= {WORKS}). The class-conditional detector is feasible; go to Phase 3.")
    elif sep.min() >= WEAK:
        head, color = 'PARTLY WORKS', '#F58518'
        text = (f"{n_good}/{n} valid runs separate clearly (AUC >= {WORKS}); the rest are between {WEAK} and "
                f"{WORKS}. Look at which model/cut/rate is weak before Phase 3.")
    else:
        bad = valid[sep < WEAK]
        where = ', '.join(f"{r.model} cut {int(r.cut)} rate {r.poison_rate:g}" for r in bad.itertuples())
        head, color = 'DOES NOT WORK EVERYWHERE', '#E45756'
        text = (f"{n_good}/{n} valid runs separate clearly, but {len(bad)} do not (AUC < {WEAK}): {where}. "
                f"Try another layer or score there, or change direction.")
    if skipped:
        text += f" {skipped} run(s) had ASR < {ASR_OK:.0f}% and are excluded from this verdict."
    weak_ac = valid[(valid['sep_auc'] >= WORKS) & (valid['ac_precision'] < 0.5)]
    if len(weak_ac):
        share = weak_ac['poison_frac_in_target_class'].min()
        text += (f" Note: in {len(weak_ac)} separable run(s) the simple 'smaller cluster = poison' rule still "
                 f"picks the wrong cluster (2-means precision < 0.5); there poison is {share:.2f}+ of the target "
                 f"class, so the clean cluster is the smaller one. The detector needs a rule that does not "
                 f"assume poison is the minority.")
    return head, color, text


def _page_summary(pdf, df, csv_path):
    fig = plt.figure(figsize=A4)
    ds = ', '.join(sorted(df['dataset'].unique()))
    fig.text(0.04, 0.95, f"Experiment 1: detector feasibility ({ds})", fontsize=17, weight='bold')
    fig.text(0.04, 0.915, "Do poisoned samples separate from clean samples inside the backdoor target class, "
             "in the smashed-data feature space?", fontsize=10.5, color='#444')
    head, color, text = _verdict(df)
    fig.text(0.04, 0.855, head, fontsize=15, weight='bold', color=color)
    fig.text(0.04, 0.835, text, fontsize=9.5, wrap=True, va='top')

    cols = ['model', 'cut', 'poison_rate', 'poison_frac_in_target_class', 'clean_acc', 'asr',
            'sep_auc', 'ss_auc', 'ac_precision', 'ac_recall']
    labels = ['Model', 'Cut', 'Poison\nrate', 'Poison share\nin class', 'Clean\nacc %', 'ASR %',
              'Separability\nAUC', 'Spectral\nAUC', '2-means\nprecision', '2-means\nrecall']
    d = df.sort_values(['model', 'cut', 'poison_rate'])
    cells, colors = [], []
    for r in d.itertuples():
        row = [r.model, int(r.cut), f"{r.poison_rate * 100:g}%", f"{r.poison_frac_in_target_class:.2f}",
               f"{r.clean_acc:.1f}", f"{r.asr:.1f}", f"{r.sep_auc:.3f}", f"{r.ss_auc:.3f}",
               f"{r.ac_precision:.2f}", f"{r.ac_recall:.2f}"]
        c = ['white'] * len(row)
        if r.asr < ASR_OK:
            c = ['#d6d6d6'] * len(row)
        else:
            c[6] = '#d9f0d3' if r.sep_auc >= WORKS else ('#fde0c5' if r.sep_auc >= WEAK else '#f8d0d0')
        cells.append(row)
        colors.append(c)
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.6])
    ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=labels, cellColours=colors, loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.55)
    for (i, _), cell in tb.get_celld().items():
        if i == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
    fig.text(0.04, 0.075,
             f"How to read: Separability AUC = how well the top principal direction splits poisoned from clean, "
             f"whichever side the poison falls on (1.0 = perfect, 0.5 = random). Green >= {WORKS}, orange "
             f"{WEAK}-{WORKS}, red < {WEAK}. Grey rows: ASR < {ASR_OK:.0f}%, backdoor not learned. Spectral AUC "
             f"(Tran et al. 2018) assumes poison is a minority of the class and fails when the poison share "
             f"approaches 0.5. 2-means flags the smaller cluster as poison (what a deployed detector would do).",
             fontsize=8, wrap=True, va='top', color='#333')
    fig.text(0.04, 0.02, f"Source: {csv_path} | generated {datetime.now():%Y-%m-%d %H:%M}", fontsize=7, color='#777')
    pdf.savefig(fig)
    plt.close(fig)


def _page_curves(pdf, df):
    fig, axes = plt.subplots(2, 2, figsize=A4)
    fig.suptitle("Detector quality and backdoor check vs poison rate", fontsize=14, weight='bold', x=0.04,
                 ha='left', y=0.97)
    groups = list(df.groupby(['model', 'cut']))
    rates = sorted(df['poison_rate'].unique())
    for k, (key, ylabel, ref) in enumerate([('sep_auc', 'Separability AUC', True),
                                            ('ss_auc', 'Spectral signature AUC', True),
                                            ('asr', 'Backdoor ASR (%)', False),
                                            ('clean_acc', 'Clean accuracy (%)', False)]):
        ax = axes.flat[k]
        for i, ((m, c), g) in enumerate(groups):
            g = g.sort_values('poison_rate')
            ax.plot(g['poison_rate'] * 100, g[key], marker='o', lw=2, color=LINE_COLORS[i % len(LINE_COLORS)],
                    label=f"{m} cut {int(c)}")
        if ref:
            ax.axhline(WORKS, ls='--', color='#54A24B', lw=1)
            ax.axhline(WEAK, ls=':', color='#E45756', lw=1)
            ax.axhline(0.5, ls='-', color='#bbbbbb', lw=0.8)
            ax.set_ylim(0.4, 1.02)
        elif key == 'asr':
            ax.axhline(ASR_OK, ls='--', color='#9D755D', lw=1)
            ax.set_ylim(0, 102)
        ax.set_xticks([r * 100 for r in rates])
        ax.set_xlabel('Poison rate (% of training set)')
        ax.set_title(ylabel, fontsize=11, loc='left')
        ax.grid(alpha=0.25)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.text(0.04, 0.015, f"Dashed green = {WORKS} (works), dotted red = {WEAK} (does not separate), grey = 0.5 "
             f"(random). ASR dashed line = {ASR_OK:.0f}% (backdoor learned).", fontsize=8, color='#333')
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _load_pca(out_dir, r):
    p = os.path.join(out_dir, f"pca_{r.dataset}_{r.model.lower()}_cut{int(r.cut)}_rate{r.poison_rate:g}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['coords'], z['poisoned'].astype(bool)


def _pages_per_run(pdf, df, out_dir, kind):
    runs = [r for r in df.sort_values(['model', 'cut', 'poison_rate']).itertuples() if _load_pca(out_dir, r)]
    if not runs:
        return
    per_page = 6
    title = ("Target-class features on the top-2 principal directions" if kind == 'scatter'
             else "Projection on the top principal direction")
    for start in range(0, len(runs), per_page):
        fig, axes = plt.subplots(2, 3, figsize=A4)
        fig.suptitle(title + " (clean vs poisoned)", fontsize=14, weight='bold', x=0.04, ha='left', y=0.97)
        for ax in axes.flat:
            ax.axis('off')
        for ax, r in zip(axes.flat, runs[start:start + per_page]):
            coords, pois = _load_pca(out_dir, r)
            ax.axis('on')
            if kind == 'scatter':
                rng = np.random.RandomState(0)
                idx = rng.permutation(len(coords))[:4000]
                c, p = coords[idx], pois[idx]
                ax.scatter(c[~p, 0], c[~p, 1], s=4, alpha=0.4, color=CLEAN, label='clean', rasterized=True)
                ax.scatter(c[p, 0], c[p, 1], s=4, alpha=0.5, color=POISON, label='poisoned', rasterized=True)
                ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
            else:
                x = coords[:, 0]
                bins = np.linspace(x.min(), x.max(), 60)
                ax.hist(x[~pois], bins=bins, alpha=0.6, color=CLEAN, label='clean')
                ax.hist(x[pois], bins=bins, alpha=0.6, color=POISON, label='poisoned')
                ax.set_xlabel('Projection on PC 1'); ax.set_ylabel('Samples')
            ax.set_title(f"{r.model} cut {int(r.cut)}, rate {r.poison_rate * 100:g}%\n"
                         f"sep AUC {r.sep_auc:.3f} | ASR {r.asr:.0f}%", fontsize=9.5)
            ax.tick_params(labelsize=7)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            ax.legend(fontsize=7, frameon=False, markerscale=3)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig, dpi=150)
        plt.close(fig)


def build_report(csv_path, out_dir=None):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows")
    out_dir = out_dir or os.path.dirname(os.path.abspath(csv_path))
    pdf_path = os.path.splitext(csv_path)[0] + '_report.pdf'
    with PdfPages(pdf_path) as pdf:
        _page_summary(pdf, df, csv_path)
        _page_curves(pdf, df)
        _pages_per_run(pdf, df, out_dir, 'scatter')
        _pages_per_run(pdf, df, out_dir, 'hist')
    return pdf_path


if __name__ == '__main__':
    paths = sys.argv[1:] or glob.glob(os.path.join('results', 'exp1_detector', 'exp1_results_*.csv'))
    if not paths:
        sys.exit("usage: python experiments/exp1_report.py <exp1_results_*.csv>")
    for p in paths:
        print(f"PDF report -> {build_report(p)}")