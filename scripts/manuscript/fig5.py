#!/usr/bin/env python3
"""Figure 5 - the tissues differ profoundly, and identically in health and disease."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_6'))
import de
import gg2
import mstyle as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import stats
import mirna_level_enrichment as E

S6 = gg2.PLOTS / 'stage_6'


def venn_counts():
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    m = m[E.paired_samples(md, m.columns)]
    s = []
    for tis in ('lens', 'tear'):
        sub, _ = de.filtered(m, md, tis)
        s.append(set(sub.index))
    a, b = s
    return len(a - b), len(a & b), len(b - a)


def panel_a(ax):
    n_lens, n_share, n_tear = venn_counts()
    ax.set_xlim(-2.05, 2.05); ax.set_ylim(-1.35, 1.35); ax.axis('off')
    for cx, col, lab in ((-.42, M.C_LENS, 'Lens'), (.42, M.C_TEAR, 'Tear')):
        ax.add_patch(Circle((cx, 0), .80, fc=col, alpha=.26, ec=col, lw=1.0))
        ax.text(cx * 2.05, .96, lab, ha='center', fontsize=8.5, fontweight='bold',
                color=col)
    for x, n, col, fs in ((-.95, n_lens, M.C_LENS, 11), (.95, n_tear, M.C_TEAR, 11),
                          (0, n_share, '#444444', 13)):
        ax.text(x, 0, str(n), ha='center', va='center', fontsize=fs,
                fontweight='bold', color=col)
    print(f'  venn: lens-only {n_lens}, shared {n_share}, tear-only {n_tear}')


def panel_b(ax):
    d = pd.read_csv(S6 / 'tissue_split_by_disease.tsv', sep='\t')
    r = stats.pearsonr(d.lfc_ctrl, d.lfc_xfg).statistic
    lim = (min(d.lfc_ctrl.min(), d.lfc_xfg.min()) - .6,
           max(d.lfc_ctrl.max(), d.lfc_xfg.max()) + .6)
    ax.plot(lim, lim, ls='--', lw=.8, color=M.GREY, zorder=1)
    ax.axhline(0, color='#e3e3e3', lw=.6, zorder=0)
    ax.axvline(0, color='#e3e3e3', lw=.6, zorder=0)
    ax.scatter(d.lfc_ctrl, d.lfc_xfg, s=4.5, c='#5f7d9a', alpha=.55, lw=0, zorder=3)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect('equal')
    ax.set_xlabel('Lens vs tear in controls, log$_2$ FC (19 pairs)')
    ax.set_ylabel('Lens vs tear in XFG, log$_2$ FC (23 pairs)')
    ax.text(.04, .96, f'$r$ = {r:.3f}\n{len(d)} miRNAs', transform=ax.transAxes,
            va='top', fontsize=7.5, fontweight='bold', color=M.INK, linespacing=1.5)
    ax.grid(alpha=.18, lw=.5); ax.set_axisbelow(True)
    print(f'  tissue contrast: r = {r:.4f}, {int((d.fdr_ctrl < .05).sum())} FDR<0.05 '
          f'in controls, {int((d.fdr_xfg < .05).sum())} in XFG, of {len(d)}')


def main():
    M.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(M.W2, M.W2 * 0.40),
                               gridspec_kw=dict(width_ratios=[.85, 1], wspace=.22,
                                                left=.03, right=.985, top=.93,
                                                bottom=.165))
    panel_a(a); panel_b(b)
    M.panel(a, 'a', dx=.0, dy=1.02)
    M.panel(b, 'b', dx=-.20)
    M.save(fig, 'figure5')


if __name__ == '__main__':
    main()
