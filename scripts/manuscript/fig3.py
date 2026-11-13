#!/usr/bin/env python3
"""Figure 3 - differential expression: nothing in lens, one candidate in tear."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import mstyle as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
from volcano_main import place_labels   # collision-aware labelling, shared with the deck

DE = gg2.PLOTS / 'stage_3' / 'de_edger.tsv'
Q = 0.05
ENTRY = 'hsa-miR-708-5p'
C_NS, C_SIG = '#b9c3cc', '#c1443f'


def bh_line(p, q=Q):
    p = np.sort(np.asarray(p, float)); n = len(p)
    n_pass = int((p <= np.arange(1, n + 1) / n * q).sum())
    return q / n, n_pass


def volcano(ax, s, cap, xlim, n_label=3):
    x = s.ql_log2FC.values
    y = -np.log10(s.ql_p.clip(lower=1e-300).values)
    crit, n_pass = bh_line(s.ql_p.values)
    yc = -np.log10(crit)
    sig = s.ql_p.values < .05
    ax.axhline(-np.log10(.05), color=M.GREY, ls=':', lw=.7, zorder=1)
    ax.axhline(yc, color='#444', ls='--', lw=.8, zorder=1)
    ax.axvline(0, color='#dddddd', lw=.6, zorder=0)
    ax.scatter(x[~sig], np.minimum(y[~sig], cap), s=4, c=C_NS, lw=0, zorder=2)
    ax.scatter(x[sig], np.minimum(y[sig], cap), s=6, c=C_SIG, lw=0, zorder=3)
    # right-edge labels: both panels' strongest miRNAs are on the left, where they
    # would collide. "adjusted P" wording per the Scientific Reports pass, 2026-11-10
    ax.text(xlim[1], yc, f'adjusted $P$ {Q:g} ', ha='right', va='bottom',
            fontsize=6.3, color='#444')
    # the P = 0.05 line goes to the LEFT edge instead: at the bottom of the panel
    # the right edge is where the hits sit, the left is empty.
    ax.text(xlim[0], -np.log10(.05), ' unadjusted $P$ 0.05', ha='left', va='top',
            fontsize=6.3, color=M.GREY)
    top = s.nsmallest(n_label, 'ql_p')
    place_labels(ax, [(r.entry, r.ql_log2FC, min(-np.log10(max(r.ql_p, 1e-300)), cap))
                      for _, r in top.iterrows()], xlim[1], cap * 1.14, 6.0)
    ax.set_xlim(*xlim); ax.set_ylim(0, cap * 1.14)
    ax.set_xlabel('log$_2$ fold change (XFG vs control)')
    ax.set_ylabel('$-$log$_{10}$ $P$')
    return n_pass


def panel_c(ax):
    md, m = gg2.metadata(), gg2.counts(gg2.PRIMARY)
    s = [c for c in m.columns if md.loc[c, 'tissue_name'] == 'tear']
    cpm = m[s] / m[s].sum() * 1e6
    y = np.log2(cpm.loc[ENTRY] + 1)
    g = md.loc[s, 'disease']
    rng = np.random.default_rng(0)
    for i, (k, lab, c) in enumerate([('control', 'Control', '#5b8c5a'),
                                     ('diseased', 'XFG', C_SIG)]):
        v = y[g == k].values
        ax.scatter(np.full(len(v), i) + rng.uniform(-.14, .14, len(v)), v, s=11,
                   color=c, alpha=.8, lw=.3, edgecolor='white', zorder=3)
        ax.plot([i - .26, i + .26], [np.median(v)] * 2, color=c, lw=1.6, zorder=4)
        print(f'  tear {lab}: n={len(v)}, median log2 CPM+1 = {np.median(v):.2f}')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Control', 'XFG'])
    ax.set_xlim(-.6, 1.6)
    ax.set_ylabel('miR-708-5p, log$_2$ (CPM + 1)')
    # pad below, then open a top band for the annotation, as Figure 2 does
    lo, hi = float(y.min()), float(y.max())
    pad = .08 * (hi - lo)
    ax.set_ylim(lo - pad, (hi + pad - (lo - pad)) / (1 - .28) + (lo - pad))
    # the point of the panel is the tension: the only miRNA to clear BH correction,
    # and distributions that plainly overlap - the numbers must be printed.
    # three lines, not two: "familywise permutation P = 0.12" on one line overruns
    # this panel's width, which is a third of the plate
    ax.text(.03, .975, 'adjusted $P$ = 0.036\nfamilywise\npermutation $P$ = 0.12',
            transform=ax.transAxes, ha='left', va='top', fontsize=6.6, color=M.INK,
            linespacing=1.45)


def main():
    M.use()
    d = pd.read_csv(DE, sep='\t')
    d = d[(d.norm == 'CPM') & (d.kind == 'relative')]
    sub = {t: d[d.tissue == t] for t in ('lens', 'tear')}
    cap = max(-np.log10(s.ql_p.min()) for s in sub.values()) * 1.10
    lim = max(abs(d.ql_log2FC).max() for s in sub.values()) * 1.08
    xlim = (-lim, lim)

    fig, axes = plt.subplots(1, 3, figsize=(M.W2, M.W2 * 0.335),
                             gridspec_kw=dict(width_ratios=[1, 1, .62], wspace=.36,
                                              left=.065, right=.99, top=.90,
                                              bottom=.185))
    for ax, t in zip(axes[:2], ('lens', 'tear')):
        n = volcano(ax, sub[t], cap, xlim)
        ax.set_title(f'{t.capitalize()} ({len(sub[t])} miRNAs)', fontsize=8, pad=3)
        print(f'  {t}: {len(sub[t])} tested, {n} at BH<{Q:g}')
    panel_c(axes[2])
    for ax, L in zip(axes, 'abc'):
        ax.grid(alpha=.16, lw=.5); ax.set_axisbelow(True)
    M.panel(axes[0], 'a', dx=-.22); M.panel(axes[1], 'b', dx=-.22)
    M.panel(axes[2], 'c', dx=-.40)
    M.save(fig, 'figure3')


if __name__ == '__main__':
    main()
