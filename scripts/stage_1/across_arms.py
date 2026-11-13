#!/usr/bin/env python3
"""Excluded entries (gg2.EXCLUDE) across all six count matrices, against known-real controls; read-level evidence in verify_suspects.py and the case studies."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

# gg2.counts() not gg2.matrix(): this figure must SHOW the excluded entries

OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
CONTROLS = ['hsa-miR-184', 'hsa-let-7a-5p', 'hsa-let-7b-5p', 'hsa-let-7c-5p',
            'hsa-miR-21-5p', 'hsa-miR-375-3p', 'hsa-miR-200c-3p', 'hsa-miR-320a-3p']


def main():
    v22, _ = gg2.refs()
    M = gg2.name_map_v22_v23()
    groups = [('INFLATED', gg2.ARTIFACTS_INFLATED),
              ('ABSENT', gg2.ARTIFACTS_ABSENT),
              ('ANNOTATION', gg2.ARTIFACTS_ANNOTATION),
              ('CONTROL (real)', [c for c in CONTROLS if c in v22])]

    bea = gg2.counts('BEA-raw')
    beac = (bea / bea.sum() * 1e6).mean(axis=1)
    names, labels, bounds = [], [], []
    for lab, ents in groups:
        ents = [e for e in ents if e in v22]
        ents.sort(key=lambda e: -float(beac.get(e, 0)))
        names += ents; labels.append(lab); bounds.append(len(names))

    data = {}
    for arm in gg2.ARMS:
        m = gg2.counts(arm); tot = m.sum(); ver = gg2.ARMS[arm][1]
        col = []
        for n in names:
            k = n if n in m.index else (M.get(n, (None,))[0] if ver == 'v23' else None)
            col.append(float((m.loc[k] / tot * 1e6).mean()) if k and k in m.index else np.nan)
        data[arm] = col
    M2 = pd.DataFrame(data, index=names)

    fig, ax = plt.subplots(figsize=(11.2, 12.4))
    im = ax.imshow(np.log10(M2.fillna(0) + .1), cmap='RdYlBu_r', aspect='auto',
                   vmin=-1, vmax=np.log10(M2.max().max() + .1))
    ax.set_xticks(range(M2.shape[1]))
    ax.set_xticklabels(M2.columns, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace('hsa-', '') for n in names], fontsize=9)
    for i in range(M2.shape[0]):
        for j in range(M2.shape[1]):
            v = M2.iloc[i, j]
            ax.text(j, i, '—' if pd.isna(v) else (f'{v:,.0f}' if v >= 1 else f'{v:.1f}'),
                    ha='center', va='center', fontsize=8,
                    color='w' if (not pd.isna(v) and v > 3000) else '#222')
    prev = 0
    for lab, b in zip(labels, bounds):
        if b < len(names):
            ax.axhline(b - .5, color='k', lw=2)
        ax.text(-.155, 1 - (prev + b) / (2 * len(names)), lab, rotation=90,
                va='center', ha='center', fontsize=10.5, weight='bold', color='#444',
                transform=ax.transAxes)
        prev = b
    ax.set_title(f'Excluded entries across all six branches\n'
                 f'mean CPM; {len(names) - len(groups[-1][1])} excluded entries above the '
                 f'last line, {len(groups[-1][1])} known-real controls below',
                 fontsize=12, weight='bold', pad=12)
    fig.colorbar(im, ax=ax, label='log10 mean CPM', fraction=.04, pad=.02)
    fig.tight_layout(rect=[.055, 0, 1, 1])
    fig.savefig(OUT / 'fig2_across_arms.png', dpi=155)
    print(f'wrote {OUT}/fig2_across_arms.png  ({len(names)} rows)')


if __name__ == '__main__':
    main()
