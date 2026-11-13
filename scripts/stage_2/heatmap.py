#!/usr/bin/env python3
"""Stage 2 — clustered heatmap of normalised miRNA counts with tissue/disease annotation (Supplementary Figure S3)."""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-dedup')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import de
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

plt.rcParams.update({'font.size': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42})
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# S2 is a manuscript figure, written to the manuscript figure set rather than this stage's plots.
OUT = gg2.ROOT / 'manuscript' / 'final_pass'
OUT.mkdir(parents=True, exist_ok=True)
C_LENS, C_TEAR = '#3a6ea5', '#c1443f'
C_CTRL, C_XFG = '#b9c4cc', '#8a4b8a'


def main():
    # Keeps the excluded sample in: this figure is the evidence for dropping it.
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY, drop_samples=False)
    keep = set()
    for tis in ['lens', 'tear']:
        mt, _ = de.filtered(m, md, tis)
        keep |= set(mt.index)
    samples = [c for c in m.columns if c in md.index]
    x = m.loc[sorted(keep), samples]
    z = np.log2(x / x.sum() * 1e6 + 1)
    z = z.loc[z.std(axis=1) > 0]
    z = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1), axis=0)

    # Euclidean + complete: R's stock hclust pipeline, the likeliest match to BEA's
    # (linkage unstated), and the conservative choice — Ward carves tidy blocks even out of noise.
    col_link = linkage(pdist(z.T.values, 'euclidean'), method='complete')
    row_link = linkage(pdist(z.values, 'euclidean'), method='complete')
    corder = dendrogram(col_link, no_plot=True)['leaves']
    rorder = dendrogram(row_link, no_plot=True)['leaves']
    zz = z.iloc[rorder, corder]
    scols = [samples[i] for i in corder]

    from scipy.cluster.hierarchy import fcluster as _fc
    _dis = np.array([md.loc[s_, 'disease'] for s_ in samples])

    def _pur(k):
        cl = _fc(col_link, k, criterion='maxclust')
        return sum(max((_dis[cl == c] == v).sum() for v in set(_dis))
                   for c in set(cl)) / len(_dis) * 100
    _d = [_pur(k) for k in range(2, 7)]
    dmin, dmax = min(_d), max(_d)

    # 180 mm two-column plate, matching the manuscript figures.
    MM = 1 / 25.4
    fig = plt.figure(figsize=(180 * MM, 102 * MM))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.5, .30, .30, .30, 9],
                          width_ratios=[40, 1], hspace=.10, wspace=.02,
                          left=.105, right=.945, top=.955, bottom=.105)

    ax = fig.add_subplot(gs[0, 0])
    dendrogram(col_link, ax=ax, no_labels=True, color_threshold=0,
               above_threshold_color='#888')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    bars = [('tissue', [C_LENS if md.loc[s, 'tissue_name'] == 'lens' else C_TEAR
                        for s in scols]),
            ('disease', [C_XFG if md.loc[s, 'disease'] == 'diseased' else C_CTRL
                         for s in scols]),
            ('QC', ['#1a1a1a' if s in gg2.EXCLUDE_SAMPLES else '#ededed'
                    for s in scols])]
    for i, (lab, colours) in enumerate(bars, start=1):
        a = fig.add_subplot(gs[i, 0])
        a.imshow([[to_rgba(c) for c in colours]], aspect='auto',
                 interpolation='nearest')
        a.set_xticks([]); a.set_yticks([])
        a.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=7,
                     labelpad=6)
        for sp in a.spines.values():
            sp.set_visible(False)

    a = fig.add_subplot(gs[4, 0])
    im = a.imshow(zz.values, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5,
                  interpolation='nearest')
    a.set_xticks([]); a.set_yticks([])
    a.set_ylabel(f'{zz.shape[0]} miRNAs, clustered', fontsize=7.5)

    cax = fig.add_subplot(gs[4, 1])
    cb = fig.colorbar(im, cax=cax); cb.set_label('z-score', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    h = [Patch(fc=C_LENS, label='lens'), Patch(fc=C_TEAR, label='tear'),
         Patch(fc=C_XFG, label='XFG'), Patch(fc=C_CTRL, label='control'),
         Patch(fc='#1a1a1a', label='excluded from all analyses (40A)')]
    fig.legend(handles=h, ncol=5, fontsize=7, frameon=False,
               loc='lower center', bbox_to_anchor=(.5, .012))

    # No title or footer: the caption carries the explanation, and internal arm names
    # never appear in a manuscript figure. PDF for typesetting as well as PNG.
    for ext, kw in (('pdf', {}), ('png', dict(dpi=400))):
        fig.savefig(OUT / f'figureS1_clustering.{ext}', facecolor='white', **kw)
    print(f'wrote {OUT}/figureS1_clustering.pdf and .png  '
          f'({zz.shape[0]} miRNAs x {len(scols)} samples, 180 x 102 mm)')

    # Purity across several cuts: with complete linkage the first cut isolates an
    # outlier group, so k=2 alone would understate the tissue separation.
    from scipy.cluster.hierarchy import fcluster
    tis = np.array([md.loc[s_, 'tissue_name'] for s_ in samples])
    dis = np.array([md.loc[s_, 'disease'] for s_ in samples])

    def purity(labels, k):
        cl = fcluster(col_link, k, criterion='maxclust')
        return sum(max((labels[cl == c] == v).sum() for v in set(labels))
                   for c in set(cl)) / len(labels)

    print('  cluster purity by number of groups:')
    for k in range(2, 7):
        print(f'    k={k}: tissue {purity(tis, k):.0%}   disease {purity(dis, k):.0%}')


if __name__ == '__main__':
    main()
