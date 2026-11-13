#!/usr/bin/env python3
"""Supplementary figure - what a five-versus-five study drawn from our own cohort reports."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
os.environ.setdefault('GG2_ARM', 'BEA-dedup')
import gg2
import mstyle as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

THEM = '#8c2f2b'
NULL = '#b9c4cc'


def main():
    M.use()
    st = gg2.PLOTS / 'stage_4'
    d = pd.read_csv(st / 'smallstudy_5v5.tsv', sep='\t')
    lfc = pd.read_csv(st / 'smallstudy_5v5_hits.tsv', sep='\t').abs_log2FC.values
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')
    their_n, their_lfc = len(theirs), theirs.logFC.abs().values

    fig, (a, b) = plt.subplots(1, 2, figsize=(M.W2, M.W2 * 0.40))

    # A
    ge = float((d.n_p10 >= their_n).mean())
    a.hist(d.n_p10, bins=np.arange(-.5, d.n_p10.max() + 2), color=NULL,
           edgecolor='white', linewidth=.3)
    a.axvline(their_n, color=THEM, lw=1.4, zorder=4)
    a.annotate(f'reported: {their_n}', (their_n, .97),
               xycoords=('data', 'axes fraction'), xytext=(5, 0),
               textcoords='offset points', color=THEM, fontsize=6.4,
               weight='bold', va='top')
    a.set_xlabel('miRNAs at $P$ < 0.1 per draw', fontsize=7)
    a.set_ylabel(f'draws (of {len(d):,})', fontsize=7)
    a.text(.97, .72, f'median {d.n_p10.median():.0f}\n{ge:.0%} returned {their_n} or more',
           transform=a.transAxes, ha='right', va='top', fontsize=6.4, color='#333',
           linespacing=1.5)
    a.grid(axis='y', alpha=.20, lw=.5); a.set_axisbelow(True)
    M.panel(a, 'a', dx=-.145, dy=1.09)

    # B
    # a few noise hits run to |log2FC| ~11; clipping them into the last bin (the slide
    # version) draws a fake density spike, so they are left off the axis and the
    # caption counts them.
    top = float(np.percentile(lfc, 99.5))
    b.hist(lfc, bins=np.linspace(0, top, 45), color=NULL,
           edgecolor='white', linewidth=.3, density=True)
    b.set_xlim(0, top)
    for v in sorted(their_lfc):
        b.axvline(v, color=THEM, lw=1.0, zorder=4)
    b.annotate(f'reported: {their_n} effects', (their_lfc.max(), .97),
               xycoords=('data', 'axes fraction'), xytext=(5, 0),
               textcoords='offset points', color=THEM, fontsize=6.4,
               weight='bold', va='top')
    b.set_xlabel('|log$_2$ fold change| of each miRNA returned', fontsize=7)
    b.set_ylabel('density', fontsize=7)
    f_small = float((lfc >= their_lfc.min()).mean())
    f_large = float((lfc >= their_lfc.max()).mean())
    b.text(.97, .72, f'{f_small:.0%} reach {their_lfc.min():.2f}, their smallest\n'
           f'{f_large:.0%} reach {their_lfc.max():.2f}, their largest',
           transform=b.transAxes, ha='right', va='top', fontsize=6.4, color='#333',
           linespacing=1.5)
    b.grid(axis='y', alpha=.20, lw=.5); b.set_axisbelow(True)
    M.panel(b, 'b', dx=-.145, dy=1.09)

    fig.tight_layout(rect=[.022, 0, 1, 1], pad=.5)
    M.save(fig, 'figureS4_smallstudy')

    as_good = float(((d.n_match >= their_n) & (d.n_p05 >= 1)).mean())
    print(f'  {len(d):,} draws, {len(lfc):,} miRNAs returned in total')
    print(f'  at P<0.1 : median {d.n_p10.median():.0f}, IQR {d.n_p10.quantile(.25):.0f}'
          f' to {d.n_p10.quantile(.75):.0f}, range {d.n_p10.min()} to {d.n_p10.max()}')
    print(f'  {ge:.0%} returned >= {their_n};  {as_good:.1%} reproduced the published '
          f'profile (>= {their_n} at |log2FC| >= {their_lfc.min():.2f}, >= 1 at P < 0.05)')
    off = int((lfc > top).sum())
    print(f'  panel B axis stops at {top:.2f}; {off} of {len(lfc):,} returned miRNAs '
          f'({off/len(lfc):.1%}) lie beyond it, max {lfc.max():.2f}')
    print(f'\n  CAPTION should carry the headline the slide version prints inside the axes:')
    print(f'   "{as_good:.0%} of draws reproduced the profile that study published."')
    print(f'   and the off-axis note: "{off} miRNAs with |log2FC| above {top:.2f} are not shown."')


if __name__ == '__main__':
    main()
