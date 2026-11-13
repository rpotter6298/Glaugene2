#!/usr/bin/env python3
"""Stage 3 — volcano plots: headline methods (edgeR QL, NBSR) each get a figure, the rest share one."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2

OUT = gg2.PLOTS / 'stage_3'
TISSUES = ['lens', 'tear']
C_UP, C_DN, C_NS, C_WEAK = '#c1443f', '#3a6ea5', '#c8c8c8', '#9fb4c7'
LFC_MIN = 1.0          # conventional 2-fold cut for what counts as 'highlighted'
NORM = 'CPM'


def load(src, **sel):
    """Return (entry, lfc, p) for one method, at CPM / relative."""
    d = pd.read_csv(OUT / src, sep='\t')
    lfc, pv = sel.pop('_lfc'), sel.pop('_p')
    if 'norm' in d.columns:
        sel['norm'] = NORM
    for k, v in sel.items():
        d = d[d[k] == v]
    d = d.rename(columns={lfc: 'lfc', pv: 'p'})
    return d[['entry', 'tissue', 'lfc', 'p']].dropna(subset=['p'])


def place_labels(ax, rows, xlim, ymax, size):
    """Label points, collisions nudged the shortest way; near the frame edge labels go inside."""
    placed = {'left': [], 'right': []}
    step = 0.045
    for entry, x, y in rows:
        outward = 'right' if x >= 0 else 'left'
        side = ('left' if x >= 0 else 'right') if abs(x) > .60 * xlim else outward
        dx = 6 if side == 'right' else -6
        fy = y / ymax
        # search outward from the point's own height for the nearest free slot
        for k in range(40):
            cand = fy + (0 if k == 0 else (k + 1) // 2 * step * (1 if k % 2 else -1))
            if 0.02 < cand < 0.99 and not any(abs(cand - q) < step for q in placed[side]):
                fy = cand
                break
        placed[side].append(fy)
        ty = fy * ymax
        moved = abs(ty - y) > ymax * .015
        kw = dict(fontsize=size, color='#333', va='center',
                  ha='left' if side == 'right' else 'right')
        if moved:
            kw.update(xytext=(x + dx * xlim / 190, ty), textcoords='data',
                      arrowprops=dict(arrowstyle='-', lw=.5, color='#aaa',
                                      shrinkA=0, shrinkB=2))
        else:
            kw.update(xytext=(dx, 0), textcoords='offset points')
        ax.annotate(entry.replace('hsa-', ''), (x, y), **kw)


def cap_for(s):
    """Ceiling for the -log10 p axis of one panel."""
    ys = np.sort(-np.log10(s.p.clip(lower=1e-300)))[::-1]
    # clip one runaway point (NBSR pins top-abundance features tightly) to a triangle at the ceiling
    cap = ys[1] * 1.25 if len(ys) > 1 and ys[0] > 3 * ys[1] else ys[0] * 1.12
    return max(cap, -np.log10(.05) * 1.6)


def bh_line(p, q=0.10):
    """Where to draw the BH line, and what passed."""
    p = np.sort(np.asarray(p, float)); n = len(p)
    n_pass = int((p <= np.arange(1, n + 1) / n * q).sum())
    bh = np.minimum.accumulate((p * n / np.arange(1, n + 1))[::-1])[::-1]
    return q / n, n_pass, float(bh[0])


def panel(ax, s, xlim, n_label=8, label_size=7.6, cap=None, bh_q=0.10):
    s = s.copy()
    s['y'] = -np.log10(s.p.clip(lower=1e-300))
    if cap is None:
        cap = cap_for(s)
    over = s.y > cap
    s['yp'] = s.y.clip(upper=cap)

    # significant AND >=2-fold; p-only points drawn pale
    sig = s.p < .05
    big = s.lfc.abs() >= LFC_MIN
    hit = sig & big
    ax.scatter(s.lfc[~sig], s.yp[~sig], s=14, c=C_NS, edgecolor='none', alpha=.7)
    ax.scatter(s.lfc[sig & ~big], s.yp[sig & ~big], s=20, c=C_WEAK,
               edgecolor='none', alpha=.85)
    for m, c in ((hit & (s.lfc > 0), C_UP), (hit & (s.lfc < 0), C_DN)):
        ax.scatter(s.lfc[m & ~over], s.yp[m & ~over], s=34, c=c,
                   edgecolor='w', linewidth=.4)
        ax.scatter(s.lfc[m & over], s.yp[m & over], s=66, c=c, marker='^',
                   edgecolor='w', linewidth=.45)
    med = s.lfc.median()
    ax.axvline(0, color='#999', lw=.8)
    for x in (-LFC_MIN, LFC_MIN):
        ax.axvline(x, color='#bbb', ls='--', lw=1)
    ax.axhline(-np.log10(.05), color='#aaa', ls=':', lw=1)
    thr, n_pass, best_q = bh_line(s.p.values, bh_q)   # counts reported to stdout
    ax.axhline(-np.log10(thr), color='#b3261e', ls='--', lw=1.2)
    ax.text(-xlim * .97, -np.log10(thr), f'  BH {bh_q:g}', fontsize=8,
            color='#b3261e', va='bottom', ha='left')

    ax.set_xlim(-xlim, xlim); ax.set_ylim(0, cap * 1.08)
    lab = s[hit].nsmallest(n_label, 'p')
    place_labels(ax, list(zip(lab.entry, lab.lfc, lab.yp)), xlim, cap * 1.08, label_size)
    ax.grid(alpha=.2); ax.set_axisbelow(True)
    return int(sig.sum()), int(hit.sum()), len(s), med, int(over.sum())


def xlim_for(frames, tis):
    """Wide enough for the bulk AND for every significant point — a hit clipped
    off the axis is the one thing a volcano must not do."""
    lo = max(np.nanpercentile(np.abs(f[f.tissue == tis].lfc), 99.5) for f in frames)
    sig = [np.abs(f[(f.tissue == tis) & (f.p < .05)].lfc).max() for f in frames]
    sig = [v for v in sig if np.isfinite(v)]
    return max([lo] + sig) * 1.18


def featured(frame, fname, suptitle, subtitle, n_label=9):
    """One method, the two tissues side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.0))
    capped = [0]
    fig.suptitle(suptitle, fontsize=16.5, weight='bold', y=.980)
    fig.text(.5, .906, subtitle, ha='center', va='center', fontsize=10.5, color='#555')
    # one y-scale for both tissues, so the panels are directly comparable
    shared_cap = max(cap_for(frame[frame.tissue == t]) for t in TISSUES)
    for ax, tis in zip(axes, TISSUES):
        nsig, nhit, ntot, med, nover = panel(ax, frame[frame.tissue == tis],
                                             xlim_for([frame], tis), n_label, 8.0,
                                             cap=shared_cap)
        capped[0] += nover
        ax.set_title(tis.capitalize(), fontsize=13, weight='bold')
        ax.set_xlabel('log2 fold change (XFG / control)', fontsize=10)
        ax.set_ylabel('−log10 p (uncorrected)', fontsize=10)
        print(f'  {tis:5s} {nsig:3d}/{ntot} at p<.05, {nhit} also |lfc|>=1')
    if capped[0]:
        fig.text(.5, .015, '▲  p below the plotted range (axis clipped so the '
                 'rest stays legible)', ha='center', fontsize=9, color='#777')
    fig.tight_layout(rect=[0, .028 if capped[0] else 0, 1, .872])
    fig.savefig(OUT / fname, dpi=150)
    print(f'wrote {OUT}/{fname}\n')


def shared(data, fname, suptitle, subtitle, n_label=5):
    """Several methods: tissues as rows, methods as columns."""
    ncol = len(data)
    fig, axes = plt.subplots(len(TISSUES), ncol, squeeze=False,
                             figsize=(4.15 * ncol + .9, 4.5 * len(TISSUES)))
    capped = [0]
    fig.suptitle(suptitle, fontsize=16.5, weight='bold', y=.978)
    fig.text(.5, .938, subtitle, ha='center', va='center', fontsize=10.5, color='#555')
    for i, tis in enumerate(TISSUES):
        xl = xlim_for([f for _, f in data], tis)
        for j, (cname, f) in enumerate(data):
            ax = axes[i][j]
            nsig, nhit, ntot, med, nover = panel(ax, f[f.tissue == tis], xl, n_label, 6.9)
            capped[0] += nover
            ax.set_title(f'{cname}\n{tis} — {nsig}/{ntot} at p<0.05, {nhit} at |log2FC|≥{LFC_MIN:g}',
                         fontsize=10.5, weight='bold')
            if j == 0:
                ax.set_ylabel('−log10 p (uncorrected)', fontsize=10)
            if i == len(TISSUES) - 1:
                ax.set_xlabel('log2 fold change (XFG / control)', fontsize=9.5)
            print(f'  {cname:16s} {tis:5s} {nsig:3d}/{ntot} at p<.05, {nhit} also |lfc|>=1')
    fig.tight_layout(rect=[0, 0, 1, .922])
    fig.savefig(OUT / fname, dpi=150)
    print(f'wrote {OUT}/{fname}\n')


def main():
    edger_ql = load('de_edger.tsv', kind='relative', _lfc='ql_log2FC', _p='ql_p')
    nbsr     = load('de_nbsr.tsv', kind='relative', _lfc='log2FC', _p='p')

    print('== RPM -> edgeR QL ==')
    featured(edger_ql, 'fig_volcano_edger.png',
             'Differential expression — edgeR',
             f'{gg2.PRIMARY}  ·  RPM, quasi-likelihood F-test, ~ sex + disease  ·  '
             'coloured = p<0.05 and |log2FC|≥1  ·  uncorrected p')

    print('== NBSR ==')
    featured(nbsr, 'fig_volcano_nbsr.png',
             'NBSR — negative-binomial softmax regression',
             f'{gg2.PRIMARY}  ·  ~ sex + disease  ·  relative (share) fold changes  ·  '
             'coloured = p<0.05 and |log2FC|≥1  ·  uncorrected p')

    print('== secondary methods ==')
    shared([('pydeseq2 Wald', load('de_results.tsv', kind='relative',
                                   _lfc='log2FoldChange', _p='pvalue')),
            ('edgeR LRT',     load('de_edger.tsv', kind='relative',
                                   _lfc='lrt_log2FC', _p='lrt_p')),
            ('limma-voom',    load('de_transform.tsv', method='voom', kind='relative',
                                   _lfc='log2FC', _p='p')),
            ('Box-Cox',       load('de_transform.tsv', method='box-cox', kind='relative',
                                   _lfc='log2FC', _p='p'))],
           'fig_volcano_secondary.png', 'Secondary methods — all on RPM',
           f'{gg2.PRIMARY}  ·  ~ sex + disease  ·  shown as a check on the two '
           'headline methods  ·  uncorrected p', n_label=5)


if __name__ == '__main__':
    main()
