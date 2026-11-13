#!/usr/bin/env python3
"""Figure 2 - unsupervised structure: tissue separates, disease does not."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_2'))
import gg2
import mstyle as M
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from de import filtered
from pca import pca, d_between

C_LENS_T, C_TEAR_T = '#c98a3f', '#3a6ea5'      # tissue colours, panel A
C_CTRL, C_CASE = '#5b8c5a', '#c1443f'          # disease colours, panels B/C


BAND = .15          # top band kept free of data so the legend never lands on a
                    # marker. Effect sizes removed from the figure 2026-10-29;
                    # main() still prints them.


def scatter(ax, P, groups, colours):
    pts = []
    for (lab, sel), c in zip(groups, colours):
        ax.scatter(P.PC1[sel], P.PC2[sel], c=c, s=11, alpha=.85, lw=.3,
                   edgecolor='white', label=f'{lab} (n = {int(sel.sum())})')
        pts.append(P.PC1[sel])
    return pts


def frame(ax, P):
    """Pad the data on all sides, then open a clean band across the top for the
    legend and panel letters; without it text landed on markers (2026-10-12)."""
    x, y = P.PC1.values, P.PC2.values
    px, py = .07 * np.ptp(x), .07 * np.ptp(y)
    x0, x1 = x.min() - px, x.max() + px
    y0, y1 = y.min() - py, y.max() + py
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y0 + (y1 - y0) / (1 - BAND))     # headroom for the band


def main():
    M.use()
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    # the SAME entry filter as the DE tests (de.filtered), not PCA's own looser
    # rule - see pca.pca's docstring for the standardisation check
    KEEP = {t: set(filtered(m, md, t)[0].index) for t in ('lens', 'tear')}
    fig, axes = plt.subplots(1, 3, figsize=(M.W2, M.W2 * 0.335),
                             gridspec_kw=dict(wspace=.34, left=.065, right=.995,
                                              top=.875, bottom=.185))

    # A
    P, var, nf = pca(m, keep=KEEP['lens'] | KEEP['tear'])
    tis = md.loc[P.index, 'tissue_name']
    ax = axes[0]
    pts = scatter(ax, P, [('lens', tis == 'lens'), ('tear', tis == 'tear')],
                  [C_LENS_T, C_TEAR_T])
    print(f'  all: PC1 d={d_between(*pts):.2f}  (not printed on the figure)')
    ax.set_xlabel(f'PC1 ({var[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1] * 100:.1f}%)')
    frame(ax, P)
    ax.set_title('All samples', fontsize=7.5, pad=3)
    ax.legend(loc='upper center', ncol=2, markerscale=1.4, handletextpad=.25,
              borderpad=.15, columnspacing=.8, handlelength=.9,
              frameon=False, fontsize=6.5, bbox_to_anchor=(.5, 1.03))

    # B, C
    for i, t in enumerate(['lens', 'tear']):
        ax = axes[i + 1]
        s_ = [c for c in m.columns if md.loc[c, 'tissue_name'] == t]
        P, var, nf = pca(m[s_], keep=KEEP[t])
        g = md.loc[P.index, 'disease']
        pts = scatter(ax, P, [('control', g == 'control'), ('XFG', g == 'diseased')],
                      [C_CTRL, C_CASE])
        d = d_between(*pts)
        p = stats.ttest_ind(*pts).pvalue
        ax.set_xlabel(f'PC1 ({var[0] * 100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var[1] * 100:.1f}%)')
        frame(ax, P)
        ax.set_title('Lens capsule' if t == 'lens' else 'Tear fluid',
                     fontsize=7.5, pad=3)
        ax.legend(loc='upper center', ncol=2, markerscale=1.4, handletextpad=.25,
                  borderpad=.15, columnspacing=.8, handlelength=.9,
                  frameon=False, fontsize=6.5, bbox_to_anchor=(.5, 1.03))
        print(f'  {t}: n={len(P)}, {nf} miRNAs, PC1 d={d:.2f} p={p:.3f}')

    for ax, L in zip(axes, 'abc'):
        ax.grid(alpha=.18, lw=.5); ax.set_axisbelow(True)
        M.panel(ax, L, dx=-.26)
    M.save(fig, 'figure2')


if __name__ == '__main__':
    main()
