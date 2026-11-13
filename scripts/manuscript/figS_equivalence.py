#!/usr/bin/env python3
"""Supplementary figure - equivalence as a field: what the data rule out, miRNA by miRNA."""
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

FC_TICKS = [1, 1.2, 1.5, 2, 3, 4, 6, 8]
EXCL = '#cfd8e0'               # the ruled-out field
KEEP = '#f6efe2'               # the field we cannot rule out
THEM = '#8c2f2b'
COL = {'lens': M.C_LENS, 'tear': M.C_TEAR}
YMAX = 3.35


def main():
    M.use()
    d = pd.read_csv(gg2.PLOTS / 'stage_4' / 'equivalence.tsv', sep='\t')
    d = d[(d.method == 'edgeR-QL') & (d['norm'] == 'CPM')]
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')

    fig, axes = plt.subplots(1, 2, figsize=(M.W2, M.W2 * 0.44), sharey=True)
    missing, stats_line = [], []
    for ax, tis, letter in zip(axes, ['lens', 'tear'], 'ab'):
        g = d[d.tissue == tis].sort_values('achieved_bound').reset_index(drop=True)
        n = len(g)
        x = np.arange(n) / (n - 1) * 100          # % of the miRNome, so n cancels
        y = g.achieved_bound.values

        ax.fill_between(x, y, YMAX, color=EXCL, zorder=1)
        ax.fill_between(x, 0, y, color=KEEP, zorder=1)
        ax.plot(x, y, color=COL[tis], lw=1.4, zorder=4)

        # the two fields are meaningless unless named; the caption does the rest
        ax.text(.035, .95, 'excluded', transform=ax.transAxes, fontsize=6.8,
                weight='bold', color='#4a5b6b', va='top', zorder=5)
        ax.text(.965, .04, 'not excluded', transform=ax.transAxes, fontsize=6.8,
                color='#8a6c2f', ha='right', va='bottom', zorder=5)

        med = float(np.median(y))
        ax.plot([0, 100], [med, med], color=COL[tis], ls=':', lw=.8, zorder=3)
        ax.text(1.5, med, f' median {2**med:.2f}x', fontsize=6.4, color=COL[tis],
                va='bottom', weight='bold', zorder=5)

        if tis == 'lens':
            pos = {e: i for i, e in enumerate(g.entry)}
            for _, r in theirs.iterrows():
                if r.miRNA not in pos:
                    missing.append(r.miRNA.replace('hsa-', ''))
                    continue
                xi, yi = pos[r.miRNA] / (n - 1) * 100, abs(r.logFC)
                ax.scatter([xi], [yi], s=26, marker='X', color=THEM,
                           edgecolor='white', linewidth=.6, zorder=6)
                # drop a line to that miRNA's own ceiling: the gap IS the claim
                ax.plot([xi, xi], [g.achieved_bound[pos[r.miRNA]], yi], color=THEM,
                        lw=.7, ls='--', alpha=.65, zorder=5)
                ha = 'left' if xi < 10 else 'right' if xi > 90 else 'center'
                ax.annotate(r.miRNA.replace('hsa-', ''), (xi, yi),
                            xytext=(6 if ha == 'left' else -6 if ha == 'right' else 0, 7),
                            textcoords='offset points', ha=ha, fontsize=5.9,
                            color=THEM, weight='bold', zorder=6)
            ax.scatter([], [], s=26, marker='X', color=THEM, edgecolor='white',
                       linewidth=.6, label='reported by Tomczyk-Socha 2022')
            ax.legend(loc='upper right', fontsize=6.4, handlelength=1.0,
                      borderpad=.3).set_zorder(7)
        else:
            e = 'hsa-miR-708-5p'
            i = g.index[g.entry == e][0]
            xi, obs, ceil = i / (n - 1) * 100, abs(g.log2FC[i]), g.achieved_bound[i]
            # rank 257/259: the marker straddles the right spine, so clip_on=False; the
            # dashed drop to its ceiling read as clutter on the near-vertical curve, so
            # the observed point alone is drawn - it lands in the not-excluded field,
            # which is the claim. main() prints the bound for the caption.
            ax.scatter([xi], [obs], s=26, marker='o', color=THEM, edgecolor='white',
                       linewidth=.6, zorder=6, clip_on=False)
            ax.annotate('miR-708-5p', (xi, obs), xytext=(-5, -1),
                        textcoords='offset points', ha='right', va='center',
                        fontsize=5.9, color=THEM, weight='bold', zorder=6)
            stats_line.append(f'  miR-708-5p: rank {i + 1}/{n}, observed '
                              f'|log2FC| {obs:.3f} ({2**obs:.2f}x), '
                              f'achieved bound {ceil:.3f} ({2**ceil:.2f}x)')

        ax.set_title(f'{"Lens capsule" if tis == "lens" else "Tear fluid"}  '
                     f'({n} miRNAs)', fontsize=7, pad=3)
        ax.set_xlim(0, 100); ax.set_ylim(0, YMAX)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('miRNAs, sorted by how tightly bounded  (%)', fontsize=7)
        ax.grid(axis='y', alpha=.22, lw=.5, zorder=0); ax.set_axisbelow(False)
        M.panel(ax, letter, dx=-.13, dy=1.10)
        stats_line.append(f'{tis}: n={n}, median bound {med:.3f} ({2**med:.2f}x), '
                          f'q90 {np.quantile(y, .9):.3f} ({2**np.quantile(y, .9):.2f}x), '
                          f'max {y.max():.2f}')

    axes[0].set_yticks([np.log2(t) for t in FC_TICKS])
    axes[0].set_yticklabels([f'{t:g}x' for t in FC_TICKS])
    axes[0].set_ylabel('effect size  (fold change, up or down)', fontsize=7)
    fig.tight_layout(pad=.4)
    M.save(fig, 'figureS3_equivalence')

    for s in stats_line:
        print(f'  {s}')
    print('\n  CAPTION must carry the untestable claim:')
    print(f'   "{", ".join(missing)} does not pass the Section 2.6 expression filter in '
          'lens capsule and\n    has no confidence interval, so five of the six reported '
          'effects are drawn in panel A."')


if __name__ == '__main__':
    main()
