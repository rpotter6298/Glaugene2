#!/usr/bin/env python3
"""Figure 1 - the data match published cohorts: before any disease comparison"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import mstyle as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

MIN_DET = 0.5          # Jones' own "core" rule: detected in >=50% of samples
ROOT = gg2.ROOT / 'plots' / 'branches'


def panel_a(ax):
    d = pd.read_csv(ROOT / 'CBGM-v22' / 'replication' / 'tear_jones_meansd.tsv',
                    sep='\t', index_col=0)
    # keep only miRNAs both cohorts actually detected
    d = d[(d.theirs_mean > 0) & (d.ours_mean > 0)]
    x, y = np.log10(d.theirs_mean), np.log10(d.ours_mean)
    core = (d.det_theirs >= MIN_DET) & (d.det_ours >= MIN_DET)
    r_all = stats.pearsonr(x, y).statistic
    r_core = stats.pearsonr(x[core], y[core]).statistic
    lim = (min(x.min(), y.min()) - .25, max(x.max(), y.max()) + .25)
    ax.plot(lim, lim, ls='--', lw=.8, color=M.GREY, zorder=1)
    ax.scatter(x[~core], y[~core], s=4, c='#b9c6d4', alpha=.65, lw=0, zorder=2,
               label=f'detected in <50% ({int((~core).sum())})')
    ax.scatter(x[core], y[core], s=5, c=M.C_TEAR, alpha=.70, lw=0, zorder=3,
               label=f'detected in \u226550% of both ({int(core.sum())})')
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect('equal')
    ax.set_xlabel('Jones et al. 2026, log$_{10}$ mean CPM')
    ax.set_ylabel('This study, log$_{10}$ mean CPM')
    ax.text(.04, .96, f'all {len(d)}:  $r$ = {r_all:.3f}\n\u226550% of both:  $r$ = {r_core:.3f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=7,
            linespacing=1.5, color=M.INK)
    ax.legend(loc='lower right', markerscale=2.0, handletextpad=.3,
              borderpad=.2, labelspacing=.25)
    ax.grid(alpha=.18, lw=.5); ax.set_axisbelow(True)


def panel_b(ax):
    """Their published top-10 vs our control-sample percentages, recomputed from the counts."""
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_top10.tsv', sep='\t')
    md = gg2.metadata()
    m = gg2.matrix('PEXlens-v22')
    lens = [c for c in m.columns if md.loc[c, 'tissue_name'] == 'lens'
            and md.loc[c, 'disease'] == 'control']
    ours_pct = m[lens].sum(axis=1) / m[lens].sum().sum() * 100
    d = theirs.copy()
    FLOOR = .003                       # below this the log axis has no room anyway
    d['ours'] = d.miRNA.map(lambda e: max(float(ours_pct.get(e, 0)), FLOOR))
    d = d.sort_values('control_pct', ascending=True).reset_index(drop=True)

    y = np.arange(len(d))
    ax.hlines(y, d.control_pct.values, d.ours.values, color='#c9d3dc', lw=1.6, zorder=1)
    ax.scatter(d.control_pct.values, y, s=13, c=M.C_LENS, zorder=3,
               label='Tomczyk-Socha 2022 (n = 5 control)')
    ax.scatter(d.ours.values, y, s=13, c=M.C_TEAR, zorder=3,
               label='This study (n = 19 control)')
    ax.set_yticks(y)
    ax.set_yticklabels([m_.replace('hsa-', '') for m_ in d.miRNA])
    ax.set_xscale('log')
    ax.set_xlabel('Share of all lens-capsule reads (%)')
    ax.set_ylim(-.7, len(d) - .3)
    for _, r in d.iterrows():
        f_ = r.control_pct / r.ours
        if f_ >= 3 or f_ <= 1 / 3:
            lab = f'{f_:.0f}×' if f_ >= 3 else f'{1 / f_:.0f}×'
            ax.annotate(lab, (max(r.ours, r.control_pct), r.name), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=6.4,
                        color='#8a6c2f')
    ax.legend(loc='upper left', markerscale=1.1, handletextpad=.3, borderpad=.2,
              labelspacing=.25)
    ax.grid(axis='x', alpha=.18, lw=.5); ax.set_axisbelow(True)


def main():
    M.use()
    fig, (a, b) = plt.subplots(1, 2, figsize=(M.W2, M.W2 * 0.44),
                               gridspec_kw=dict(width_ratios=[1, 1.05], wspace=.42,
                                                left=.085, right=.985, top=.93,
                                                bottom=.155))
    panel_a(a); M.panel(a, 'a', dx=-.20)
    panel_b(b); M.panel(b, 'b', dx=-.30)
    M.save(fig, 'figure1')


if __name__ == '__main__':
    main()
