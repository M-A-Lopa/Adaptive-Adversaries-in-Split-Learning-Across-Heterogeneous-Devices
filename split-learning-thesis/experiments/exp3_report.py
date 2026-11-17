"""
PDF report for Experiment 3 (client-level backdoor detection in multi-client split learning).

Called automatically at the end of exp3_client_defence.py.  To rebuild it:
    python experiments/exp3_report.py results/exp3_client/exp3_results_CIFAR10.csv

Pages
  1. verdict + table: ASR and clean accuracy per method, which client was flagged and when
  2. ASR vs poison rate (rows = models, columns = cuts, one line per method)
  3. clean accuracy vs poison rate
  4+ detector signal per round in the undefended runs: malicious client's z-score vs the
     highest honest client, one page per poison rate
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

ASR_LEARNED = 80.0      # undefended ASR needed for the configuration to count
ASR_STOPPED = 20.0      # defended ASR at or below this = backdoor stopped
MAX_ACC_DROP = 3.0      # clean accuracy may fall by at most this many points
A4 = (11.69, 8.27)
ORDER = ['none', 'detect', 'detect_cum', 'detect_both', 'oracle']
COLORS = {'none': '#8c8c8c', 'detect': '#4C78A8', 'detect_cum': '#B279A2', 'detect_both': '#F58518',
          'oracle': '#54A24B'}
LABELS = {'none': 'no defence', 'detect': 'detector (single round z)',
          'detect_cum': 'detector (cumulative score)',
          'detect_both': 'detector (cumulative + self-change score)',
          'oracle': 'oracle: client removed when it starts poisoning (upper bound)'}
KEY = ['model', 'cut', 'poison_rate']


def _prepare(df):
    base = df[df['method'] == 'none'][KEY + ['asr', 'clean_acc']].rename(
        columns={'asr': 'asr_none', 'clean_acc': 'acc_none'})
    m = df.merge(base, on=KEY, how='left')
    m['acc_drop'] = m['acc_none'] - m['clean_acc']
    m['stopped'] = (m['asr'] <= ASR_STOPPED) & (m['acc_drop'] <= MAX_ACC_DROP)
    return m


def _main_method(m):
    have = set(m['method'])
    for meth in ('detect_both', 'detect_cum', 'detect'):
        if meth in have:
            return meth
    return 'detect'


def _verdict(m):
    main = _main_method(m)
    det = m[m['method'] == main]
    pois = det[(det['poison_rate'] > 0) & (det['asr_none'] >= ASR_LEARNED)]
    weak = det[(det['poison_rate'] > 0) & (det['asr_none'] < ASR_LEARNED)]
    clean = det[det['poison_rate'] == 0]
    if pois.empty:
        return ('NOT ENOUGH RESULTS', '#9D755D',
                "No defended run yet where the undefended backdoor reached ASR >= "
                f"{ASR_LEARNED:.0f}%.", [])
    n, stopped = len(pois), int(pois['stopped'].sum())
    found = int(pois['malicious_flagged'].astype(bool).sum())
    delay = pois['detection_delay'].dropna()
    ff_pois = int((pois['false_flags'] > 0).sum())
    ff_clean = int((clean['false_flags'] > 0).sum())
    share = stopped / n
    if share >= 0.9:
        head, color = 'DEFENCE WORKS', '#54A24B'
    elif share >= 0.5:
        head, color = 'DEFENCE PARTLY WORKS', '#F58518'
    else:
        head, color = 'DEFENCE DOES NOT WORK YET', '#E45756'
    text = (f"{LABELS[main]}: backdoor stopped in {stopped}/{n} poisoned configurations (ASR <= {ASR_STOPPED:.0f}% and clean "
            f"accuracy within {MAX_ACC_DROP:.0f} points of no defence).")
    orc = m[(m['method'] == 'oracle') & (m['poison_rate'] > 0) & (m['asr_none'] >= ASR_LEARNED)]
    lines = [
        f"Mean ASR: no defence {pois['asr_none'].mean():.1f}%  ->  detector {pois['asr'].mean():.1f}%"
        + (f"  (oracle upper bound {orc['asr'].mean():.1f}%)" if len(orc) else ''),
        f"Malicious client identified in {found}/{n} configurations"
        + (f"; median delay {delay.median():.0f} round(s) after poisoning started (max {delay.max():.0f})"
           if len(delay) else ''),
        f"False alarms: an honest client was flagged in {ff_pois}/{n} poisoned runs and in "
        f"{ff_clean}/{len(clean)} clean (0% poison) runs",
        f"Mean clean-accuracy change vs no defence: {-pois['acc_drop'].mean():+.2f} points",
    ]
    for other in ('detect_cum', 'detect'):
        if other != main and other in set(m['method']) and ORDER.index(other) < ORDER.index(main):
            old = m[(m['method'] == other) & (m['poison_rate'] > 0) & (m['asr_none'] >= ASR_LEARNED)]
            lines.append(f"For comparison, {LABELS[other]}: stopped {int(old['stopped'].sum())}/{len(old)}, "
                         f"mean ASR {old['asr'].mean():.1f}%")
    if len(weak):
        lines.append(f"{len(weak)} poisoned configuration(s) left out: the undefended backdoor stayed below "
                     f"{ASR_LEARNED:.0f}% ASR there (nothing to stop)")
    return head, color, text, lines


def _page_summary(pdf, m, csv_path):
    fig = plt.figure(figsize=A4)
    ds = ', '.join(sorted(m['dataset'].unique()))
    r0 = m.iloc[0]
    fig.text(0.04, 0.955, f"Experiment 3: client-level backdoor detection ({ds})", fontsize=15, weight='bold')
    fig.text(0.04, 0.928, f"{int(r0['clients'])} clients, client {int(r0['malicious'])} malicious from round "
             f"{int(r0['poison_start'])}, {int(r0['rounds'])} rounds, target class {int(r0['target'])}, "
             f"flag at robust z >= {r0['z_thr']:g}"
             + (f", data: {m['data'].dropna().iloc[-1]}" if 'data' in m and m['data'].notna().any() else ", data: IID"),
             fontsize=9, color='#444')
    head, color, text, lines = _verdict(m)
    fig.text(0.04, 0.89, head, fontsize=14, weight='bold', color=color)
    fig.text(0.04, 0.862, text, fontsize=9.5)
    for k, ln in enumerate(lines):
        fig.text(0.05, 0.838 - 0.022 * k, '- ' + ln, fontsize=8.5)

    piv = m.pivot_table(index=KEY, columns='method', values=['asr', 'clean_acc'], aggfunc='first')
    main = _main_method(m)
    det = m[m['method'] == main].set_index(KEY)
    meths = [x for x in ORDER if x in set(m['method'])]
    short = {'none': 'no def.', 'detect': 'detect', 'detect_cum': 'det. cum', 'detect_both': 'det. both',
             'oracle': 'oracle'}
    labels = ['Model', 'Cut', 'Poison\n(total)'] + [f"ASR\n{short[x]}" for x in meths] + \
             [f"Acc\n{short[x]}" for x in meths] + [f'Flagged\n{short[main]}', 'Delay\n(rounds)', 'False\nflags']
    cells, colors = [], []
    for key in piv.index:
        model, cut, rate = key
        g = lambda v, meth: piv.loc[key].get((v, meth), np.nan)
        f = lambda x, nd=1: '-' if pd.isna(x) else f"{x:.{nd}f}"
        d = det.loc[key] if key in det.index else None
        row = [model, int(cut), f"{rate * 100:g}%"] + [f(g('asr', x)) for x in meths] + \
              [f(g('clean_acc', x)) for x in meths] + [
               '-' if d is None or pd.isna(d['flagged']) or not str(d['flagged']).strip() else str(d['flagged']),
               '-' if d is None or pd.isna(d['detection_delay']) else f"{d['detection_delay']:.0f}",
               '-' if d is None else f"{int(d['false_flags'])}"]
        col = ['white'] * len(row)
        a0 = g('asr', 'none')
        for j, x in enumerate(meths):
            a = g('asr', x)
            if x == 'none' or rate == 0 or pd.isna(a):
                continue
            if not pd.isna(a0) and a0 < ASR_LEARNED:
                col[3 + j] = '#e6e6e6'
            else:
                col[3 + j] = '#d9f0d3' if a <= ASR_STOPPED else ('#fde0c5' if a <= 50 else '#f8d0d0')
            acc_d, acc_0 = g('clean_acc', x), g('clean_acc', 'none')
            if not pd.isna(acc_d) and not pd.isna(acc_0) and acc_0 - acc_d > MAX_ACC_DROP:
                col[3 + len(meths) + j] = '#f8d0d0'
        if d is not None and d['false_flags'] > 0:
            col[-1] = '#f8d0d0'
        cells.append(row)
        colors.append(col)
    top = 0.838 - 0.022 * len(lines) - 0.015
    ax = fig.add_axes([0.02, 0.075, 0.96, top - 0.075])
    ax.axis('off')
    tb = ax.table(cellText=cells, colLabels=labels, cellColours=colors, loc='upper center', cellLoc='center',
                  bbox=[0, 0, 1, 1])            # fit the table inside the axes so it never runs into the footnote
    tb.auto_set_font_size(False)
    tb.set_fontsize(6 if len(cells) > 24 else 7.5)
    for (i, _), cell in tb.get_celld().items():
        if i == 0:
            cell.set_facecolor('#e8e8e8')
            cell.set_text_props(weight='bold')
    fig.text(0.04, 0.012, f"ASR of a defence: green <= {ASR_STOPPED:.0f}% (stopped), orange <= 50%, red > 50%, grey = "
             f"undefended backdoor below {ASR_LEARNED:.0f}%. Acc of a defence red = more than {MAX_ACC_DROP:.0f} points "
             f"below no defence. Delay = rounds between the start of poisoning and the flag. "
             f"Source: {csv_path} | {datetime.now():%Y-%m-%d %H:%M}", fontsize=7, color='#444', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def _grid(pdf, m, value, ylabel, title, ylim=None, ref=None, skip_zero=False):
    models, cuts = sorted(m['model'].unique()), sorted(m['cut'].unique())
    fig, axes = plt.subplots(len(models), len(cuts), figsize=A4, squeeze=False, sharey=True)
    fig.suptitle(title, fontsize=14, weight='bold', x=0.04, ha='left', y=0.975)
    for i, mo in enumerate(models):
        for j, cu in enumerate(cuts):
            ax = axes[i][j]
            g = m[(m['model'] == mo) & (m['cut'] == cu)]
            if skip_zero:
                g = g[g['poison_rate'] > 0]
            for meth in [x for x in ORDER if x in set(g['method'])]:
                h = g[g['method'] == meth].sort_values('poison_rate')
                ax.plot(h['poison_rate'] * 100, h[value], marker='o', lw=2, color=COLORS[meth], label=LABELS[meth])
            if ref is not None:
                ax.axhline(ref, ls='--', color='#54A24B', lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"{mo}, cut {int(cu)}", fontsize=10)
            ax.set_xticks(sorted(g['poison_rate'].unique() * 100))
            ax.grid(alpha=0.25)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if i == len(models) - 1:
                ax.set_xlabel('Poison rate (% of all training data)')
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labs = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='lower center', ncol=len(labs), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def _pages_signal(pdf, rd, m):
    """Undefended runs: z-score of the malicious client vs the highest honest client, per round."""
    if rd is None or rd.empty:
        return
    g0 = rd[rd['method'] == 'none']
    zcols = [c for c in g0.columns if c.startswith('z_c')]
    if g0.empty or not zcols:
        return
    r0 = m.iloc[0]
    mal, start, thr = int(r0['malicious']), int(r0['poison_start']), float(r0['z_thr'])
    malcol = f'z_c{mal}'
    others = [c for c in zcols if c != malcol]
    for rate in sorted(g0['poison_rate'].unique()):
        g = g0[g0['poison_rate'] == rate]
        models, cuts = sorted(g['model'].unique()), sorted(g['cut'].unique())
        fig, axes = plt.subplots(len(models), len(cuts), figsize=A4, squeeze=False, sharey=True)
        what = "clean run (no poison): every client should stay below the threshold" if rate == 0 else \
            f"poison {rate * 100:g}% of all data, starting in round {start}"
        fig.suptitle(f"Detector signal per round, no defence - {what}", fontsize=12.5, weight='bold', x=0.04,
                     ha='left', y=0.975)
        for i, mo in enumerate(models):
            for j, cu in enumerate(cuts):
                ax = axes[i][j]
                h = g[(g['model'] == mo) & (g['cut'] == cu)].sort_values('round')
                h = h.dropna(subset=[malcol])
                if not h.empty:
                    ax.plot(h['round'], h[malcol].clip(lower=0.1), marker='o', color='#E45756', lw=2,
                            label=f"client {mal} (malicious)" if rate > 0 else f"client {mal}")
                    ax.plot(h['round'], h[others].max(axis=1).clip(lower=0.1), marker='s', color='#8c8c8c', lw=1.5,
                            label='highest honest client' if rate > 0 else 'highest other client')
                ax.axhline(thr, ls='--', color='#4C78A8', lw=1.2, label=f'flag threshold (z = {thr:g})')
                if rate > 0:
                    ax.axvline(start - 0.5, ls=':', color='#555', lw=1)
                ax.set_yscale('log')
                ax.set_title(f"{mo}, cut {int(cu)}", fontsize=10)
                ax.grid(alpha=0.25, which='both')
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
                if i == len(models) - 1:
                    ax.set_xlabel('Round')
                if j == 0:
                    ax.set_ylabel('Robust z-score (log scale)')
        handles, labs = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labs, loc='lower center', ncol=3, frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.02))
        if rate > 0:
            fig.text(0.04, 0.005, "Dotted vertical line: poisoning starts.", fontsize=8, color='#444')
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)


def build_report(csv_path, rounds_csv=None):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no rows")
    if rounds_csv is None:
        rounds_csv = csv_path.replace('exp3_results_', 'exp3_rounds_')
    rd = pd.read_csv(rounds_csv) if os.path.exists(rounds_csv) else None
    m = _prepare(df)
    pdf_path = os.path.splitext(csv_path)[0] + '_report.pdf'
    with PdfPages(pdf_path) as pdf:
        _page_summary(pdf, m, csv_path)
        _grid(pdf, m, 'asr', 'ASR (%)', 'Backdoor ASR vs poison rate (lower is better)', ylim=(0, 102),
              ref=ASR_STOPPED, skip_zero=True)
        _grid(pdf, m, 'clean_acc', 'Clean accuracy (%)', 'Clean accuracy vs poison rate (higher is better)')
        _pages_signal(pdf, rd, m)
    return pdf_path


if __name__ == '__main__':
    paths = sys.argv[1:] or glob.glob(os.path.join('results', 'exp3_client', 'exp3_results_*.csv'))
    if not paths:
        sys.exit("usage: python experiments/exp3_report.py <exp3_results_*.csv>")
    for p in paths:
        print(f"PDF report -> {build_report(p)}")