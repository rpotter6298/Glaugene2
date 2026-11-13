#!/usr/bin/env python3
"""Supplementary figure - no recorded covariate structures the within-tissue projection."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_2'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
os.environ.setdefault('GG2_ARM', 'BEA-dedup')
import gg2
import mstyle as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from de import filtered
from pca import pca, d_between
from scipy import stats

# label, column, the level drawn as "present"
BINARY = [('Disease (XFG)', 'disease', 'diseased'), ('Male sex', 'sex', 1),
          ('Diabetes', 'diabetes', 1), ('Hypertension', 'hypertension', 1),
          ('Dementia', 'dementia', 1), ('Smoker', 'smoke', 1),
          ('Vitamin supplement', 'vitamin', 1), ('Family history', 'family_history', 1)]
CONT = [('Age', 'age'), ('Schirmer score', 'schirmer'), ('Cigarettes per day', 'cigs_per_day')]
C_NO, C_YES = '#9fb3c4', '#c1443f'


def projections():
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    out = {}
    for t in ('lens', 'tear'):
        cols = [c for c in m.columns if md.loc[c, 'tissue_name'] == t]
        keep = set(filtered(m, md, t)[0].index)
        P, var, nf = pca(m[cols], keep=keep)
        out[t] = (P, var, nf, md.loc[cols], m[cols].sum())
    return out


def stat_rows(proj):
    rows = []
    for t in ('lens', 'tear'):
        P, _, _, md_t, depth = proj[t]
        for lab, col, pos in BINARY:
            v = md_t[col]
            g1 = P.PC1[v[v == pos].index]
            g0 = P.PC1[v[v != pos].index] if col == 'disease' else \
                 P.PC1[v[v.fillna(0) != pos].index]
            rows.append(dict(tissue=t, label=lab, kind='d',
                             stat=d_between(g1, g0),
                             p=stats.ttest_ind(g1, g0).pvalue, n=f'{len(g1)}/{len(g0)}'))
        for lab, col in CONT:
            v = pd.to_numeric(md_t[col], errors='coerce')
            ok = v.notna()
            r = stats.spearmanr(P.PC1[ok.values], v[ok])
            rows.append(dict(tissue=t, label=lab, kind='rho',
                             stat=abs(r.statistic), p=r.pvalue, n=int(ok.sum())))
        r = stats.spearmanr(P.PC1, np.log10(depth))
        rows.append(dict(tissue=t, label='Library depth', kind='rho',
                         stat=abs(r.statistic), p=r.pvalue, n=len(depth)))
    return pd.DataFrame(rows)


def scatter_grid(fig, gs, proj, t, letter, title):
    P, var, nf, md_t, _ = proj[t]
    for i, (lab, col, pos) in enumerate(BINARY):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        v = md_t[col]
        yes = v == pos if col == 'disease' else v.fillna(0) == pos
        ax.scatter(P.PC1[~yes], P.PC2[~yes], s=5, c=C_NO, lw=0, alpha=.85)
        ax.scatter(P.PC1[yes], P.PC2[yes], s=5, c=C_YES, lw=0, alpha=.85)
        ax.set_title(f'{lab}  ({int(yes.sum())}/{int((~yes).sum())})', fontsize=6.4, pad=2.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_linewidth(.5); sp.set_color('#cccccc')
        if i == 0:
            ax.set_ylabel('PC2', fontsize=6.4)
            M.panel(ax, letter, dx=-.34, dy=1.60)
            ax.text(-.12, 1.62, f'{title}  ({nf} miRNAs, PC1 {100*var[0]:.1f}%)',
                    transform=ax.transAxes, fontsize=7.4, fontweight='bold',
                    va='bottom', ha='left', color=M.INK)
        if i // 4 == 1:
            ax.set_xlabel('PC1', fontsize=6.4)


def main():
    M.use()
    proj = projections()
    st = stat_rows(proj)
    fig = plt.figure(figsize=(M.W2, M.W2 * 0.58))
    outer = fig.add_gridspec(2, 1, hspace=.60, left=.075, right=.985,
                             top=.885, bottom=.075)
    scatter_grid(fig, outer[0].subgridspec(2, 4, hspace=.40, wspace=.16),
                 proj, 'lens', 'a', 'Lens capsule')
    scatter_grid(fig, outer[1].subgridspec(2, 4, hspace=.40, wspace=.16),
                 proj, 'tear', 'b', 'Tear fluid')
    M.save(fig, 'figureS2_covariates')

    print('\n  association with PC1 (asterisk = nominal P < 0.05):')
    for t in ('lens', 'tear'):
        sct = st[st.tissue == t]
        print(f'   {t}: {int((sct.p < .05).sum())} of {len(sct)} at P < 0.05  '
              f'| strongest: {sct.loc[sct.stat.idxmax(), "label"]} '
              f'{sct.stat.max():.2f} (P {sct.loc[sct.stat.idxmax(), "p"]:.3f})')
        cov = sct[sct.label != 'Library depth']
        print(f'        excluding depth: {int((cov.p < .05).sum())} of {len(cov)}; '
              f'closest P = {cov.p.min():.3f} ({cov.loc[cov.p.idxmin(), "label"]})')
    return

    # ---- dropped summary panel, kept for reference ----
    ax = fig.add_subplot(outer[2])
    labs = [l for l, _, _ in BINARY] + [l for l, _ in CONT] + ['Library depth']
    y = np.arange(len(labs))[::-1]
    h = .30
    for off, t, col in ((h / 2, 'lens', M.C_LENS), (-h / 2, 'tear', M.C_TEAR)):
        s = st[st.tissue == t].set_index('label').reindex(labs)
        ax.barh(y + off, s.stat.values, h, color=col, label=f'{t} (n = '
                f'{len(proj[t][0])})', zorder=3)
        for yy, v, p in zip(y + off, s.stat.values, s.p.values):
            ax.text(v + .02, yy, f'{v:.2f}' + ('*' if p < .05 else ''), va='center',
                    fontsize=5.9, color='#444')
    ax.axhline(len(CONT) + .5, color='#999999', lw=.7, ls=':')
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=6.8)
    ax.set_xlabel("Association with the first principal component: Cohen's $d$ above the "
                  "dotted line, |Spearman $\\rho$| below", fontsize=7)
    ax.set_xlim(0, max(st.stat) * 1.22)
    ax.legend(loc='lower right', fontsize=6.6, handlelength=1.0, borderpad=.25)
    ax.grid(axis='x', alpha=.18, lw=.5); ax.set_axisbelow(True)
    M.panel(ax, 'c', dx=-.275, dy=1.08)
    M.save(fig, 'figureS2_covariates')

    print('\n  association with PC1 (asterisk = nominal P < 0.05):')
    for t in ('lens', 'tear'):
        s = st[st.tissue == t]
        print(f'   {t}: {int((s.p < .05).sum())} of {len(s)} at P < 0.05  '
              f'| strongest: {s.loc[s.stat.idxmax(), "label"]} '
              f'{s.stat.max():.2f} (P {s.loc[s.stat.idxmax(), "p"]:.3f})')
        cov = s[s.label != 'Library depth']
        print(f'        excluding depth: {int((cov.p < .05).sum())} of {len(cov)}; '
              f'closest P = {cov.p.min():.3f} ({cov.loc[cov.p.idxmin(), "label"]})')


if __name__ == '__main__':
    main()
