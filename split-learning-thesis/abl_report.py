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
DEFENSE_TITLE = {'splitfss': 'SplitFSS (Make Split, not Hijack)', 'zorro': 'ZORRO',
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


def build_pdf(rows, path, meta):
    df = pd.DataFrame(rows)
    if df.empty:
        return None
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
    models = list(dict.fromkeys(df['model'].tolist()))
    story = [Paragraph('Ablation study: which defense component stops which attack', st['Title']),
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
    doc.build(story)
    return path


def _build_matplotlib(df, path, meta):
    from matplotlib.backends.backend_pdf import PdfPages
    models = list(dict.fromkeys(df['model'].tolist()))
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.text(0.05, 0.92, 'Ablation study: which defense component stops which attack', fontsize=16, weight='bold')
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
    return path
