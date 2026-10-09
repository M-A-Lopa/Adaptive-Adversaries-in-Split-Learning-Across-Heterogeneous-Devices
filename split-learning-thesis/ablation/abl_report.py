import io
import math
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRIC_LABEL = {'ssim': 'SSIM (lower = better defense)', 'auc': 'Leak AUC (lower = better defense)',
                'asr': 'ASR % (lower = better defense)'}
VERDICT_COLOR = {'RESPONSIBLE': '#c8e6c9', 'partial': '#fff3c4', 'not responsible': '#eeeeee',
                 'removal helps': '#ffd7d7', 'reference': '#dde7f7'}
DEFAULT_TITLE = 'Ablation study: which defense component stops which attack'
DEFENSE_TITLE = {'triguard': 'TriGuard-SL (combined defense)',
                 'protoguard_paper': 'ProtoGuard-SL (paper-faithful)',
                 'splitfss': 'SplitFSS (Make Split, not Hijack)', 'zorro': 'ZORRO',
                 'splitml': 'SplitML', 'r3elu': 'R3eLU (Secure Split Learning)',
                 'protoguard': 'ProtoGuard-SL', 'securesplit': 'SecureSplit'}


def _fmt(v, metric):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    return f"{v:.2f}" if metric == 'asr' else f"{v:.4f}"


def _num(v, nd=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    return f"{v:.{nd}f}"


def _verdict(r):
    return r['verdict'] if r.get('status') == 'ok' else str(r.get('status', ''))[:24]


def responsible_summary(df):
    out = []
    for (d, a), g in df.groupby(['defense', 'attack'], sort=False):
        row = {'defense': d, 'attack': a}
        for m, gm in g.groupby('model', sort=False):
            gate = gm['gate_substantial'].iloc[0]
            comps = gm[gm['verdict'] == 'RESPONSIBLE']['variant'].tolist()
            part = gm[gm['verdict'] == 'partial']['variant'].tolist()
            if not gate:
                txt = 'defense not substantial: skipped'
            else:
                txt = ', '.join(comps) if comps else 'none >= 0.5'
                if part:
                    txt += f" (partial: {', '.join(part)})"
            row[m] = txt
        out.append(row)
    return out


def chart_png(g, models, metric, title):
    variants = [v for v in g['variant'].unique() if v not in ('full', 'off')]
    if not variants:
        return None
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(variants) + 1.4))
    y = np.arange(len(variants))
    h = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        gm = g[g['model'] == m].set_index('variant')
        vals = [gm.loc[v, 'responsibility'] if v in gm.index and pd.notna(gm.loc[v, 'responsibility']) else 0.0
                for v in variants]
        ax.barh(y + i * h, vals, height=h, label=m)
    ax.axvline(0.5, color='green', ls='--', lw=1)
    ax.axvline(0.2, color='orange', ls=':', lw=1)
    ax.axvline(0.0, color='black', lw=0.8)
    ax.set_yticks(y + h * (len(models) - 1) / 2)
    ax.set_yticklabels(variants, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('responsibility = share of the protection lost when the component is removed')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf, fig.get_size_inches()


def attack_table(g, models, metric):
    variants = list(dict.fromkeys(g['variant'].tolist()))
    head = ['Variant']
    for m in models:
        head += [f'{m} {metric}', f'{m} acc%', f'{m} resp', f'{m} verdict']
    rows = [head]
    for v in variants:
        r = [v]
        for m in models:
            s = g[(g['model'] == m) & (g['variant'] == v)]
            if s.empty:
                r += ['-', '-', '-', '-']
                continue
            s = s.iloc[0].to_dict()
            r += [_fmt(s.get('value'), metric), _num(s.get('acc')), _num(s.get('responsibility')), _verdict(s)]
        rows.append(r)
    return rows


SWEEP_ATTACKS = [('whitebox', 'White-box SSIM'), ('ae_decoder', 'AE decoder SSIM'),
                 ('fsha', 'FSHA SSIM'), ('label_leakage', 'Label-leak AUC')]
SWEEP_TEXT = ("R3eLU is reported first at the paper's budget (epsilon = 1, global sensitivity 2KC). In this "
              "thesis setting the client holds the whole image and the server holds no features of its own, so "
              "every bit of signal the server learns from passes through the R3eLU-noised cut layer. With "
              "K = half the cut-layer features and C = 10, the Laplace scale 2KC/epsilon is orders of magnitude "
              "larger than the clipped activations, and the model falls to chance accuracy. The paper evaluates a "
              "vertical split in which the host also holds its own un-noised half of the features, which is why it "
              "reports usable accuracy at the same epsilon. This setting difference, not an implementation error, "
              "explains the gap. The sweep below increases epsilon to show where R3eLU starts to keep usable "
              "accuracy in this setting and how much protection is left at that point. Usable means within "
              "{tol:g} accuracy points of the undefended model. Protection kept = (no-defense metric - metric at "
              "that epsilon) / (no-defense metric - metric at the smallest epsilon), so 100% means the full "
              "protection of the strongest setting is still present and 0% means none is left.")


def load_sweep(path):
    import os
    if not path or not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict('records')


def sweep_analysis(sdf, tol):
    out = {}
    for m, g in sdf.groupby('model', sort=False):
        g = g[g['status'] == 'ok'] if 'status' in g else g
        off = g[g['variant'] == 'off']
        eps_rows = g[g['variant'] != 'off']
        eps_list = sorted(eps_rows['epsilon'].dropna().unique().tolist())
        def val(sub, a):
            r = sub[sub['attack'] == a]
            return float(r['value'].iloc[0]) if not r.empty and pd.notna(r['value'].iloc[0]) else None
        def acc(sub):
            a = sub['acc'].dropna()
            return float(a.iloc[0]) if not a.empty else None
        off_acc = acc(off)
        table = [{'label': 'No defense', 'eps': None, 'acc': off_acc, **{a: val(off, a) for a, _ in SWEEP_ATTACKS}}]
        for e in eps_list:
            sub = eps_rows[eps_rows['epsilon'] == e]
            table.append({'label': f"epsilon = {e:g}", 'eps': e, 'acc': acc(sub),
                          **{a: val(sub, a) for a, _ in SWEEP_ATTACKS}})
        usable = None
        if off_acc is not None:
            for row in table[1:]:
                if row['acc'] is not None and row['acc'] >= off_acc - tol:
                    usable = row
                    break
        kept = {}
        if usable is not None and len(table) > 1:
            strongest = table[1]
            for a, _ in SWEEP_ATTACKS:
                o, v, s0 = table[0][a], usable[a], strongest[a]
                if None in (o, v, s0) or (o - s0) <= 0.02:
                    kept[a] = None
                else:
                    kept[a] = max(0.0, min(1.0, (o - v) / (o - s0)))
        out[m] = {'table': table, 'usable': usable, 'kept': kept, 'off_acc': off_acc}
    return out


def sweep_chart_png(model, info):
    rows = [r for r in info['table'] if r['eps'] is not None]
    if not rows:
        return None
    eps = [r['eps'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    ax = axes[0]
    ax.plot(eps, [r['acc'] if r['acc'] is not None else np.nan for r in rows], marker='o', label='R3eLU')
    if info['off_acc'] is not None:
        ax.axhline(info['off_acc'], color='grey', ls='--', lw=1, label='no defense')
    ax.set_xscale('log')
    ax.set_xlabel('epsilon (log scale)')
    ax.set_ylabel('test accuracy %')
    ax.set_title(f'{model}: utility', fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    for i, (a, lab) in enumerate(SWEEP_ATTACKS):
        ys = [r[a] if r[a] is not None else np.nan for r in rows]
        if all(np.isnan(ys)):
            continue
        line, = ax.plot(eps, ys, marker='o', label=lab)
        if info['table'][0][a] is not None:
            ax.axhline(info['table'][0][a], color=line.get_color(), ls=':', lw=1)
    if info['usable'] is not None:
        ax.axvline(info['usable']['eps'], color='green', ls='--', lw=1)
        axes[0].axvline(info['usable']['eps'], color='green', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('epsilon (log scale)')
    ax.set_ylabel('attack success (higher = more leakage)')
    ax.set_title(f'{model}: privacy (dotted = no defense)', fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf, fig.get_size_inches()


def sweep_rows_table(info):
    head = ['Setting', 'Accuracy %'] + [lab for _, lab in SWEEP_ATTACKS]
    rows = [head]
    for r in info['table']:
        rows.append([r['label'], _num(r['acc'])] +
                    [_fmt(r[a], 'asr' if a == 'asr' else 'ssim') for a, _ in SWEEP_ATTACKS])
    return rows


def sweep_sentence(model, info):
    if info['off_acc'] is None:
        return f"{model}: no undefended reference available."
    if info['usable'] is None:
        return (f"{model}: no tested epsilon keeps accuracy within the tolerance of the undefended model "
                f"({info['off_acc']:.2f}%). R3eLU never reaches usable accuracy in this range.")
    u = info['usable']
    parts = []
    for a, lab in SWEEP_ATTACKS:
        k = info['kept'].get(a)
        if k is not None:
            parts.append(f"{lab} {_fmt(u[a], 'ssim')} vs {_fmt(info['table'][0][a], 'ssim')} undefended "
                         f"({100 * k:.0f}% of the protection kept)")
    return (f"{model}: first usable at epsilon = {u['eps']:g} (accuracy {u['acc']:.2f}% vs "
            f"{info['off_acc']:.2f}% undefended). At that point: " + ('; '.join(parts) if parts else
                                                                     'no attack metric available') + '.')


def headline_rows(df, models):
    head = ['Attack', 'Metric']
    for m in models:
        head += [f'{m}: no defense', f'{m}: defended', f'{m}: acc %']
    rows = [head]
    for a in dict.fromkeys(df['attack'].tolist()):
        g = df[df['attack'] == a]
        metric = g['metric'].iloc[0]
        r = [a, metric]
        for m in models:
            gm = g[g['model'] == m]
            off = gm[gm['variant'] == 'off']
            full = gm[gm['variant'] == 'full']
            ov = off['value'].iloc[0] if not off.empty else None
            fv = full['value'].iloc[0] if not full.empty else None
            fa = full['acc'].iloc[0] if not full.empty and 'acc' in full else None
            r += [_fmt(ov, metric), _fmt(fv, metric), _num(fa)]
        rows.append(r)
    return rows


def _headline_story(df, models, st, small, width):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
    data = headline_rows(df, models)
    n = len(data[0])
    t = Table(data, colWidths=[3.0 * cm, 1.6 * cm] + [(width - 4.6 * cm) / (n - 2)] * (n - 2), repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                           ('GRID', (0, 0), (-1, -1), 0.3, colors.grey), ('ALIGN', (2, 1), (-1, -1), 'CENTER')]))
    return [Spacer(1, 8), Paragraph('Headline: no defense vs. full defense', st['Heading2']), t, Spacer(1, 8)]


def build_pdf(rows, path, meta):
    df = pd.DataFrame(rows)
    sweep = pd.DataFrame(meta.get('sweep_rows') or [])
    if df.empty and sweep.empty:
        return None
    if not df.empty:
        df = df.drop_duplicates(subset=['defense', 'model', 'dataset', 'attack', 'variant'], keep='last')
    if not sweep.empty:
        sweep = sweep.drop_duplicates(subset=['model', 'variant', 'attack'], keep='last')
    meta = dict(meta, sweep_df=sweep)
    try:
        return _build_reportlab(df, path, meta)
    except ImportError:
        return _build_matplotlib(df, path, meta)


def _build_reportlab(df, path, meta):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                    PageBreak, KeepTogether)

    st = getSampleStyleSheet()
    small = ParagraphStyle('small', parent=st['Normal'], fontSize=8, leading=10)
    cell = ParagraphStyle('cell', parent=st['Normal'], fontSize=7, leading=8.5)
    doc = SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=1.3 * cm, rightMargin=1.3 * cm,
                            topMargin=1.2 * cm, bottomMargin=1.2 * cm,
                            title='Ablation study of split-learning defenses')
    width = landscape(A4)[0] - 2.6 * cm
    sweep = meta.get('sweep_df')
    if df.empty:
        story = [Paragraph(meta.get('title', DEFAULT_TITLE), st['Title'])]
        story += _sweep_story(sweep, meta, st, small, width)
        doc.build(story)
        return path
    models = list(dict.fromkeys(df['model'].tolist()))
    story = [Paragraph(meta.get('title', DEFAULT_TITLE), st['Title']),
             Paragraph(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} | models: {', '.join(models)} | "
                       f"dataset: {', '.join(df['dataset'].unique())} | cut layer: {meta.get('cut')} | "
                       f"seed: {meta.get('seed')}", small),
             Spacer(1, 6),
             Paragraph("<b>How to read.</b> For every defense and attack, <i>full</i> is the complete defense and "
                       "<i>off</i> is the same pipeline with the defense disabled. Every other row removes or "
                       "replaces one component. <b>Responsibility</b> = (metric with component removed - metric of "
                       "full) / (metric of off - metric of full): 1.0 means removing the component gives the "
                       "attacker back all of the protection, 0 means the component does not matter, negative means "
                       "the defense works better without it. Verdicts: RESPONSIBLE >= 0.5, partial >= 0.2. "
                       f"Component rows are only run when the full defense gives a substantial change "
                       f"(SSIM drop >= {meta['gates']['ssim']}, leak AUC drop >= {meta['gates']['auc']}, "
                       f"ASR drop >= {meta['gates']['asr']} points); otherwise they are marked as skipped.", small),
             Spacer(1, 6),
             Paragraph(f"Budgets: {meta.get('budgets')}", small),
             Spacer(1, 10),
             Paragraph('Summary: responsible components per attack', st['Heading2'])]
    if meta.get('intro'):
        story[2:2] = [Spacer(1, 4)] + [Paragraph(t, small) for t in meta['intro']]
    if meta.get('headline'):
        idx = len(story) - 1
        story[idx:idx] = _headline_story(df, models, st, small, width)

    summ = responsible_summary(df)
    data = [['Defense', 'Attack'] + models]
    for s in summ:
        data.append([Paragraph(DEFENSE_TITLE.get(s['defense'], s['defense']), cell), s['attack']] +
                    [Paragraph(s.get(m, '-'), cell) for m in models])
    cw = [4.2 * cm, 2.6 * cm] + [(width - 6.8 * cm) / max(len(models), 1)] * len(models)
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
                           ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                           ('FONTSIZE', (0, 0), (-1, -1), 7.5), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
                           ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])]))
    story.append(t)

    for d in dict.fromkeys(df['defense'].tolist()):
        story.append(PageBreak())
        story.append(Paragraph(DEFENSE_TITLE.get(d, d), st['Heading1']))
        gd = df[df['defense'] == d]
        for a in dict.fromkeys(gd['attack'].tolist()):
            g = gd[gd['attack'] == a]
            metric = g['metric'].iloc[0]
            block = [Paragraph(f"Attack: {a} &nbsp; | &nbsp; metric: {METRIC_LABEL.get(metric, metric)}", st['Heading3'])]
            gaps = []
            for m in models:
                gm = g[g['model'] == m]
                if not gm.empty:
                    gp = gm['protection_gap'].iloc[0]
                    gaps.append(f"{m}: protection gap {_num(gp, 4 if metric != 'asr' else 2)} "
                                f"({'substantial' if gm['gate_substantial'].iloc[0] else 'not substantial'})")
            block.append(Paragraph(' | '.join(gaps), small))
            block.append(Spacer(1, 4))
            data = attack_table(g, models, metric)
            n = len(data[0])
            cw = [5.2 * cm] + [(width - 5.2 * cm) / (n - 1)] * (n - 1)
            t = Table(data, colWidths=cw, repeatRows=1)
            style = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
                     ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTSIZE', (0, 0), (-1, -1), 7),
                     ('GRID', (0, 0), (-1, -1), 0.3, colors.grey), ('ALIGN', (1, 1), (-1, -1), 'CENTER')]
            for i, r in enumerate(data[1:], start=1):
                for j in range(4, n, 4):
                    c = VERDICT_COLOR.get(r[j])
                    if c:
                        style.append(('BACKGROUND', (j, i), (j, i), colors.HexColor(c)))
            t.setStyle(TableStyle(style))
            block.append(t)
            ch = chart_png(g, models, metric, f"{DEFENSE_TITLE.get(d, d)} vs {a}")
            if ch is not None and g['responsibility'].notna().any():
                buf, (w_in, h_in) = ch
                w = min(width * 0.75, w_in * 72)
                block += [Spacer(1, 4), Image(buf, width=w, height=w * h_in / w_in)]
            block.append(Spacer(1, 12))
            story.append(KeepTogether(block[:4]))
            story += block[4:]
    story += _sweep_story(sweep, meta, st, small, width)
    doc.build(story)
    return path


def _sweep_story(sweep, meta, st, small, width):
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    if sweep is None or sweep.empty:
        return []
    tol = meta.get('usable_tol', 5.0)
    info = sweep_analysis(sweep, tol)
    story = [PageBreak(), Paragraph('R3eLU privacy-utility sweep over epsilon', st['Heading1']),
             Paragraph(SWEEP_TEXT.format(tol=tol), small), Spacer(1, 8)]
    for m, inf in info.items():
        story.append(Paragraph(f"Model: {m}", st['Heading2']))
        story.append(Paragraph(sweep_sentence(m, inf), small))
        story.append(Spacer(1, 4))
        data = sweep_rows_table(inf)
        n = len(data[0])
        t = Table(data, colWidths=[4.0 * cm] + [(width - 4.0 * cm) / (n - 1)] * (n - 1), repeatRows=1)
        style = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
                 ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('FONTSIZE', (0, 0), (-1, -1), 7.5),
                 ('GRID', (0, 0), (-1, -1), 0.3, colors.grey), ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
                 ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#dde7f7'))]
        if inf['usable'] is not None:
            idx = 1 + [r['eps'] for r in inf['table'][1:]].index(inf['usable']['eps']) + 1
            style.append(('BACKGROUND', (0, idx), (-1, idx), colors.HexColor('#c8e6c9')))
        t.setStyle(TableStyle(style))
        story.append(t)
        ch = sweep_chart_png(m, inf)
        if ch is not None:
            buf, (w_in, h_in) = ch
            w = min(width, w_in * 72)
            story += [Spacer(1, 4), Image(buf, width=w, height=w * h_in / w_in)]
        story.append(Spacer(1, 10))
    return story


def _build_matplotlib(df, path, meta):
    from matplotlib.backends.backend_pdf import PdfPages
    models = list(dict.fromkeys(df['model'].tolist())) if not df.empty else []
    with PdfPages(path) as pdf:
        _matplotlib_main(pdf, df, meta, models)
        sweep = meta.get('sweep_df')
        if sweep is not None and not sweep.empty:
            tol = meta.get('usable_tol', 5.0)
            for m, inf in sweep_analysis(sweep, tol).items():
                data = sweep_rows_table(inf)
                fig = plt.figure(figsize=(11.69, 8.27))
                fig.text(0.03, 0.95, f"R3eLU epsilon sweep - {m}", fontsize=12, weight='bold')
                fig.text(0.03, 0.91, sweep_sentence(m, inf)[:230], fontsize=7)
                ax = fig.add_axes([0.03, 0.55, 0.94, 0.33])
                ax.axis('off')
                tb = ax.table(cellText=data[1:], colLabels=data[0], loc='upper left', cellLoc='center')
                tb.auto_set_font_size(False)
                tb.set_fontsize(7)
                pdf.savefig(fig)
                plt.close(fig)
    return path


def _matplotlib_main(pdf, df, meta, models):
    if df.empty:
        return
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.92, meta.get('title', DEFAULT_TITLE), fontsize=16, weight='bold')
    fig.text(0.05, 0.87, f"models: {', '.join(models)} | dataset: {', '.join(df['dataset'].unique())} | "
                         f"cut {meta.get('cut')} | seed {meta.get('seed')}", fontsize=9)
    summ = responsible_summary(df)
    cells = [[s['defense'], s['attack']] + [s.get(m, '-')[:70] for m in models] for s in summ]
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.78])
    ax.axis('off')
    if cells:
        tb = ax.table(cellText=cells, colLabels=['defense', 'attack'] + models, loc='upper left', cellLoc='left')
        tb.auto_set_font_size(False)
        tb.set_fontsize(7)
        tb.scale(1, 1.3)
    pdf.savefig(fig)
    plt.close(fig)
    for (d, a), g in df.groupby(['defense', 'attack'], sort=False):
        metric = g['metric'].iloc[0]
        data = attack_table(g, models, metric)
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.03, 0.95, f"{DEFENSE_TITLE.get(d, d)} vs {a} ({METRIC_LABEL.get(metric, metric)})",
                 fontsize=12, weight='bold')
        ax = fig.add_axes([0.02, 0.05, 0.96, 0.86])
        ax.axis('off')
        tb = ax.table(cellText=data[1:], colLabels=data[0], loc='upper left', cellLoc='center')
        tb.auto_set_font_size(False)
        tb.set_fontsize(6.5)
        tb.scale(1, 1.25)
        pdf.savefig(fig)
        plt.close(fig)