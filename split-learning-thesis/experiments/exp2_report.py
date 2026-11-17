"""
PDF report for Experiment 2 (detector in the training loop vs the client backdoor).

Called automatically at the end of exp2_detector_defence.py.  To rebuild it:
    python experiments/exp2_report.py results/exp2_defence/exp2_results_CIFAR10.csv

Pages
  1. verdict + table: ASR and clean-accuracy cost per rule, per model / cut / poison rate
  2. ASR vs poison rate (rows = models, columns = cuts, one line per rule)
  3. clean-accuracy cost and share of clean training data wrongly removed
  4. detector signal over training: target-class silhouette vs the highest other class
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

ASR_LEARNED = 80.0      # undefended ASR needed for the run to count
ASR_STOPPED = 20.0      # defended ASR at or below this = backdoor stopped
MAX_ACC_DROP = 3.0      # clean accuracy may fall by at most this many points
A4 = (11.69, 8.27)
COLORS = {'none': '#8c8c8c', 'smaller': '#F58518', 'tighter': '#4C78A8', 'crossclass': '#B279A2', 'oracle': '#54A24B'}
LABELS = {'none': 'no defence', 'smaller': 'detector: smaller cluster', 'tighter': 'detector: tighter cluster',
          'crossclass': 'detector: cross-class', 'oracle': 'detector: oracle (upper bound)'}


def _prepare(df):
    base = df[df['rule'] == 'none'][['model', 'cut', 'poison_rate', 'asr', 'clean_acc']]
    base = base.rename(columns={'asr': 'asr_none', 'clean_acc': 'acc_none'})
    m = df.merge(base, on=['model', 'cut', 'poison_rate'], how='left')
    m['acc_drop'] = m['acc_none'] - m['clean_acc']
    m['stopped'] = (m['asr'] <= ASR_STOPPED) & (m['acc_drop'] <= MAX_ACC_DROP)
    return m


def _verdict(m):
    rules = [r for r in ['smaller', 'tighter', 'crossclass', 'oracle'] if r in set(m['rule'])]
    pois = m[(m['poison_rate'] > 0) & (m['rule'] != 'none') & (m['asr_none'] >= ASR_LEARNED)]
    if pois.empty or not rules:
        return 'NOT ENOUGH RESULTS', '#9D755D', "No defended poisoned runs with a learned backdoor yet.", []
    lines = []
    best, best_key, best_n, n_cfg = None, None, -1, 0
    for r in rules:
        g = pois[pois['rule'] == r]
        n_cfg = len(g)
        n = int(g['stopped'].sum())
        clean = m[(m['rule'] == r) & (m['poison_rate'] == 0)]
        fp = clean['clean_removed_pct'].mean() if len(clean) else float('nan')
        lines.append(f"{LABELS[r]}: backdoor stopped in {n}/{len(g)} poisoned configs "
                     f"(ASR <= {ASR_STOPPED:.0f}% and accuracy drop <= {MAX_ACC_DROP:.0f} pt); "
                     f"mean ASR {g['asr'].mean():.1f}% (undefended {g['asr_none'].mean():.1f}%); "
                     f"clean data removed in clean runs: {fp:.2f}%")
        key = (n, -g['asr'].mean())       # most configs stopped, then lowest mean ASR
        if r != 'oracle' and (best_key is None or key > best_key):
            best, best_key, best_n = r, key, n
    share = best_n / max(n_cfg, 1)
    if share >= 0.9:
        head, color = 'DEFENCE WORKS', '#54A24B'
    elif share >= 0.5:
        head, color = 'DEFENCE PARTLY WORKS', '#F58518'
    else:
        head, color = 'DEFENCE DOES NOT WORK YET', '#E45756'
    text = (f"Best deployable rule: {LABELS.get(best, best)}, stopping the backdoor in {best_n}/{n_cfg} "
            f"poisoned configurations.")
    return head, color, text, lines


def _page_summary(pdf, m, csv_path):
    fig = plt.figure(figsize=A4)
    ds = ', '.join(sorted(m['dataset'].unique()))
    tg = ', '.join(str(int(t)) for t in sorted(m['target'].unique())) if 'target' in m else '0'
    fig.text(0.04, 0.955, f"Experiment 2: detector in training vs client backdoor ({ds}, target class {tg})",
             fontsize=15, weight='bold')
    head, color, text, lines = _verdict(m)
    fig.text(0.04, 0.905, head, fontsize=14, weight='bold', color=color)
    fig.text(0.04, 0.875, text, fontsize=9.5)
    for k, ln in enumerate(lines):
        fig.text(0.05, 0.85 - 0.022 * k, '- ' + ln, fontsize=8)

    rules = [r for r in ['smaller', 'tighter', 'crossclass', 'oracle'] if r in set(m['rule'])]
    piv = m.pivot_table(index=['model', 'cut', 'poison_rate'], columns='rule',
                        values=['asr', 'acc_drop', 'clean_removed_pct'], aggfunc='first')
    base = m[m['rule'] == 'none'].set_index(['model', 'cut', 'poison_rate'])
    labels = ['Model', 'Cut', 'Poison', 'ASR\nno def.', 'Acc\nno def.']
    for r in rules:
        labels += [f'ASR\n{r}', f'Acc drop\n{r}', f'Clean rem.\n{r} %']
    cells, colors = [], []
    for key in piv.index:
        model, cut, rate = key
        b = base.loc[key] if key in base.index else None
        row = [model, int(cut), f"{rate * 100:g}%",
               f"{b['asr']:.1f}" if b is not None else '-', f"{b['clean_acc']:.1f}" if b is not None else '-']
        col = ['white'] * 5
        for r in rules:
            a = piv.loc[key].get(('asr', r), np.nan)
            d = piv.loc[key].get(('acc_drop', r), np.nan)
            c = piv.loc[key].get(('clean_removed_pct', r), np.nan)
            row += ['-' if pd.isna(a) else f"{a:.1f}", '-' if pd.isna(d) else f"{d:+.1f}",
                    '-' if pd.isna(c) else f"{c:.2f}"]
            if rate == 0 or pd.isna(a):
                ca = 'white'
            else:
                ca = '#d9f0d3' if a <= ASR_STOPPED else ('#fde0c5' if a <= 50 else '#f8d0d0')
            cd = 'white' if pd.isna(d) else ('#f8d0d0' if d > MAX_ACC_DROP else 'white')
            col += [ca, cd, 'white']
        cells.append(row)
        colors.append(col)
    top = 0.85 - 0.022 * len(lines) - 0.02
    ax = fig.add_axes([0.02, 0.06, 0.96, top - 0.06])
    ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=labels, cellColours=colors, loc='upper center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5 if len(cells) > 24 else 7.5)
    tb.scale(1, 1.0 if len(cells) > 24 else 1.3)
    for (i, _), cell in tb.get_celld().items():
        if i == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
    fig.text(0.04, 0.03, f"Green ASR <= {ASR_STOPPED:.0f}% (stopped), orange <= 50%, red > 50%. Acc drop = clean "
             f"accuracy lost vs no defence (red if > {MAX_ACC_DROP:.0f} pt). Clean rem. = share of clean training "
             f"samples the detector dropped in the last epoch. Source: {csv_path} | {datetime.now():%Y-%m-%d %H:%M}",
             fontsize=7, color='#444', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _grid(pdf, m, value, ylabel, title, ylim=None, ref=None, skip_zero=False):
    models = sorted(m['model'].unique())
    cuts = sorted(m['cut'].unique())
    rules = [r for r in RULE_ORDER if r in set(m['rule'])]
    fig, axes = plt.subplots(len(models), len(cuts), figsize=A4, squeeze=False, sharey=True)
    fig.suptitle(title, fontsize=14, weight='bold', x=0.04, ha='left', y=0.975)
    for i, mo in enumerate(models):
        for j, cu in enumerate(cuts):
            ax = axes[i][j]
            g = m[(m['model'] == mo) & (m['cut'] == cu)]
            if skip_zero:
                g = g[g['poison_rate'] > 0]
            for r in rules:
                h = g[g['rule'] == r].sort_values('poison_rate')
                if h.empty or h[value].isna().all():
                    continue
                ax.plot(h['poison_rate'] * 100, h[value], marker='o', lw=2, color=COLORS[r], label=LABELS[r])
            if ref is not None:
                ax.axhline(ref, ls='--', color='#54A24B', lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"{mo}, cut {int(cu)}", fontsize=10)
            ax.set_xticks(sorted(g['poison_rate'].unique() * 100))
            ax.grid(alpha=0.25)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if i == len(models) - 1:
                ax.set_xlabel('Poison rate (%)')
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=len(labs), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


RULE_ORDER = ['none', 'smaller', 'tighter', 'crossclass', 'oracle']


def _page_signal(pdf, ep, m):
    """Detector signal at the last detection of each run, against the poison rate."""
    if ep is None or ep.empty or 'target_silhouette' not in ep:
        return
    rule = 'crossclass' if 'crossclass' in set(ep['rule']) else next((r for r in ep['rule'].unique() if r != 'none'), None)
    if rule is None:
        return
    g0 = ep[ep['rule'] == rule].dropna(subset=['target_silhouette'])
    if g0.empty:
        return
    last = g0.sort_values('epoch').groupby(['model', 'cut', 'poison_rate']).tail(1)
    models = sorted(last['model'].unique())
    cuts = sorted(last['cut'].unique())
    fig, axes = plt.subplots(len(models), len(cuts), figsize=A4, squeeze=False, sharey=True)
    fig.suptitle(f"Detector signal at detection time (rule {rule}): silhouette of the target class vs the "
                 f"highest other class", fontsize=12.5, weight='bold', x=0.04, ha='left', y=0.975)
    for i, mo in enumerate(models):
        for j, cu in enumerate(cuts):
            ax = axes[i][j]
            h = last[(last['model'] == mo) & (last['cut'] == cu)].sort_values('poison_rate')
            if not h.empty:
                x = h['poison_rate'] * 100
                ax.plot(x, h['target_silhouette'], marker='o', color='#E45756', lw=2, label='target class (0)')
                ax.plot(x, h['other_silhouette_max'], marker='s', color='#8c8c8c', lw=1.5,
                        label='highest other class')
                ax.plot(x, h['threshold'], marker='_', ls='--', color='#4C78A8', lw=1.2,
                        label='threshold (suspicious above)')
                ax.set_xticks(list(x))
            ax.set_title(f"{mo}, cut {int(cu)}", fontsize=10)
            ax.grid(alpha=0.25)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if i == len(models) - 1:
                ax.set_xlabel('Poison rate (%)')
            if j == 0:
                ax.set_ylabel('Silhouette')
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.025))
    fig.text(0.04, 0.005, "At 0% poison the target class should stay below the threshold (no false alarm); "
             "with poison it should rise above it.", fontsize=8, color='#444')
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def build_report(csv_path, epochs_csv=None):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows")
    if 'target' not in df:
        df['target'] = 0
    if epochs_csv is None:
        epochs_csv = csv_path.replace('exp2_results_', 'exp2_epochs_')
    ep = pd.read_csv(epochs_csv) if os.path.exists(epochs_csv) else None
    m = _prepare(df)
    pdf_path = os.path.splitext(csv_path)[0] + '_report.pdf'
    with PdfPages(pdf_path) as pdf:
        _page_summary(pdf, m, csv_path)
        _grid(pdf, m, 'asr', 'ASR (%)', 'Backdoor ASR vs poison rate (lower is better)', ylim=(0, 102),
              ref=ASR_STOPPED, skip_zero=True)
        d = m[m['rule'] != 'none']
        if not d.empty:
            _grid(pdf, d, 'acc_drop', 'Accuracy drop (pt)', 'Clean-accuracy cost of the detector (vs no defence)',
                  ref=MAX_ACC_DROP)
            _grid(pdf, d, 'clean_removed_pct', 'Clean data removed (%)',
                  'Clean training samples wrongly dropped in the last epoch (0% poison = false positives)')
        _page_signal(pdf, ep, m)
    return pdf_path


if __name__ == '__main__':
    paths = sys.argv[1:] or glob.glob(os.path.join('results', 'exp2_defence', 'exp2_results_*.csv'))
    paths = [p for p in paths if not p.endswith('_epochs.csv')]
    if not paths:
        sys.exit("usage: python experiments/exp2_report.py <exp2_results_*.csv>")
    for p in paths:
        print(f"PDF report -> {build_report(p)}")