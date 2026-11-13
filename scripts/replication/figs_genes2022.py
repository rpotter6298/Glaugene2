#!/usr/bin/env python3
"""Replication B - the three figures that walk through the Genes 2022 comparison (composition, abundance, claims vs our CIs)."""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'PEXlens-v22')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = gg2.PLOTS / 'replication'
OUT.mkdir(parents=True, exist_ok=True)
# C reads/writes the PRIMARY arm, not this script's PEXlens-v22 arm: keeping it
# under PEXlens-v22 is what let the old table be mistaken for a replication-arm result
PRIMARY_ARM = 'BEA-dedup'
OUT_P = gg2.SHARED / 'branches' / PRIMARY_ARM / 'replication'
CPM_MIN, FRAC = 3.7, 0.50
C_THEM, C_US, C_BAD = '#c1443f', '#3a6ea5', '#e0a33c'


def lens_data():
    md = gg2.metadata()
    m = gg2.counts(gg2.PRIMARY)
    lens = [c for c in m.columns if md.loc[c, 'tissue_name'] == 'lens']
    return m[lens], md.loc[lens]


# ---------------------------------------------------------------- A
def fig_composition(t, md):
    """Their published top-10 against ours, plus our lens/tear abundance profiles vs the external tear reference."""
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_top10.tsv', sep='\t')
    ctl = [c for c in t.columns if md.loc[c, 'disease'] == 'control']
    ours = t[ctl].sum(axis=1) / t[ctl].sum().sum() * 100
    d = theirs.copy()
    d['ours'] = d.miRNA.map(lambda e: max(float(ours.get(e, 0)), .003))
    d = d.sort_values('control_pct', ascending=True)
    ratio = d.ours / d.control_pct
    off = (ratio > 3) | (ratio < 1 / 3)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.4, 6.4),
                                 gridspec_kw={'width_ratios': [1, 1.02], 'wspace': .26})
    y = np.arange(len(d))
    ax.hlines(y, d.control_pct, d.ours, color='#c8d2da', lw=2.4, zorder=1)
    ax.scatter(d.control_pct, y, s=78, c=C_THEM, zorder=3,
               label='Genes 2022 (n=5 control)', edgecolor='w', linewidth=.8)
    ax.scatter(d.ours, y, s=78, c=C_US, zorder=3,
               label=f'GlauGene2 lens (n={len(ctl)} control)', edgecolor='w', linewidth=.8)
    for i, (_, r) in enumerate(d.iterrows()):
        if off.iloc[i]:
            ax.annotate(f'{r.ours / r.control_pct:.0f}x' if r.ours > r.control_pct
                        else f'{r.control_pct / r.ours:.0f}x lower',
                        (max(r.ours, r.control_pct), i), xytext=(9, 0),
                        textcoords='offset points', fontsize=8.6, va='center',
                        color='#8a6c2f')
    ax.set_yticks(y)
    ax.set_yticklabels([e.replace('hsa-', '') for e in d.miRNA], fontsize=10)
    for lbl, bad in zip(ax.get_yticklabels(), off):
        lbl.set_color('#8a6c2f' if bad else '#16202b')
    ax.set_xscale('log'); ax.set_xlim(.004, 200)
    ax.set_xlabel('share of all lens reads (%)', fontsize=10.5)
    top5 = ours.sort_values(ascending=False)
    ax.set_title('A — Genes 2022 Top 10', fontsize=12.5, weight='bold', loc='left')
    ax.grid(axis='x', alpha=.25); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9.5, loc='center left',
              bbox_to_anchor=(.01, .72))

    # ---- right half: abundance profile ----
    m_all = gg2.counts(gg2.PRIMARY)
    meta = gg2.metadata()
    curves = []
    for tis, col in [('lens', C_US), ('tear', '#c1443f')]:
        cols = [c for c in m_all.columns
                if c in meta.index and meta.loc[c, 'tissue_name'] == tis]
        cpm = m_all[cols].sum(axis=1) / m_all[cols].sum().sum() * 1e6
        curves.append((f'GlauGene2 {tis} (n={len(cols)})', cpm, col, '-'))
    j = pd.read_csv(gg2.ROOT / 'refs' / 'GSE291021_jones2026_tear_counts.csv.gz',
                    index_col=0)
    jc = j.sum(axis=1) / j.sum().sum() * 1e6
    curves.append((f'Jones 2026 tear (n={j.shape[1]}, external)', jc, '#8a6c2f', '--'))

    for lab, cpm, col, ls in curves:
        v = np.sort(cpm[cpm > 0].values)[::-1]
        bx.plot(np.arange(1, len(v) + 1), v, ls, color=col, lw=2.2, label=lab)
    lens_cpm = curves[0][1]
    hit = [(e, lens_cpm.get(e, 0)) for e in theirs.miRNA if lens_cpm.get(e, 0) > 0]
    rank = {e: int((lens_cpm > v).sum()) + 1 for e, v in hit}
    bx.scatter([rank[e] for e, _ in hit], [v for _, v in hit], s=46, zorder=5,
               facecolor='none', edgecolor=C_THEM, linewidth=1.5,
               label='their 10, where they sit in ours')
    bx.set_xscale('log'); bx.set_yscale('log')
    bx.set_xlabel('miRNA rank (most abundant first)', fontsize=10.5)
    bx.set_ylabel('mean CPM', fontsize=10.5)
    bx.set_title('B — Abundance Profile', fontsize=12.5, weight='bold', loc='left')
    # the Genes 2022 row is summed from the per-miRNA percentages they publish
    # (61.9 + 16.6 + 5.5 + 4.2 + 1.7 = 89.9 top-5); they state no concentration figure
    conc = {}
    for lab, cpm, _c, _ls in curves:
        v = np.sort(cpm.values)[::-1] / 1e4
        conc[lab.split(' (')[0]] = (v[:5].sum(), v[:10].sum())
    tp = theirs.control_pct.sort_values(ascending=False)
    conc['Genes 2022 lens'] = (tp.head(5).sum(), tp.sum())
    rows = [(lab, *conc[key]) for key, lab in
            [('Jones 2026 tear', 'Jones 2026 tear'),
             ('GlauGene2 tear', 'GlauGene2 tear'),
             ('Genes 2022 lens', 'Genes 2022 lens'),
             ('GlauGene2 lens', 'GlauGene2 lens')]]
    tbl = f'{"":<17}{"top-5":>7}{"top-10":>8}\n' + '\n'.join(
        f'{k:<17}{a:>6.0f}%{b:>7.0f}%' for k, a, b in rows)
    bx.text(.975, .97, tbl, transform=bx.transAxes, ha='right', va='top',
            fontsize=8.6, color='#444', family='monospace', linespacing=1.5,
            bbox=dict(fc='white', ec='#ddd', alpha=.92))
    bx.grid(alpha=.25, which='both'); bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=9, loc='lower left')

    fig.tight_layout()
    fig.savefig(OUT / 'fig_lens_composition.png', dpi=150, facecolor='white')
    print(f'A wrote fig_lens_composition.png  ({int(off.sum())} entries off by >3x)')


# ---------------------------------------------------------------- B
def fig_abundance(t, md):
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')
    cpm = t / t.sum() * 1e6
    med = cpm.median(axis=1)
    med = med[med > 0]
    claimed = theirs[theirs.meets_logfc_criterion == 'yes'].miRNA.tolist()

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.hist(np.log10(med), bins=46, color='#c8d2da', edgecolor='w', linewidth=.6)
    ax.axvline(np.log10(CPM_MIN), color='#555', ls='--', lw=1.2)
    ax.text(np.log10(CPM_MIN), ax.get_ylim()[1] * .97, '  their 3.7 CPM filter',
            fontsize=9, color='#555', va='top')
    ymax = ax.get_ylim()[1]
    for i, e in enumerate(claimed):
        v = cpm.loc[e].median() if e in cpm.index else 0
        lab = e.replace('hsa-', '')
        if v <= 0:
            ax.annotate(f'{lab}\nmedian 0 CPM\n(zero in '
                        f'{int((t.loc[e] == 0).sum())}/{t.shape[1]} samples)',
                        xy=(np.log10(med.min()), ymax * .40), fontsize=9,
                        color=C_BAD, ha='left', weight='bold')
            continue
        ax.axvline(np.log10(v), color=C_THEM, lw=1.5, alpha=.85)
        ax.annotate(lab, (np.log10(v), ymax * (.86 - .085 * i)), xytext=(6, 0),
                    textcoords='offset points', fontsize=9.2, color=C_THEM,
                    weight='bold', va='center')
    ax.set_xlabel('median expression across our lens samples, log10 CPM', fontsize=10.5)
    ax.set_ylabel('number of miRNAs', fontsize=10.5)
    ax.set_title('Where their four reported miRNAs sit in the lens miRNome',
                 fontsize=13.5, weight='bold', pad=30)
    ax.text(.5, 1.012, 'their Figure 2 plots all four on an axis of 0-30 normalised '
            'counts; let-7b-5p needed 0-15,000',
            transform=ax.transAxes, ha='center', fontsize=10, color='#555')
    ax.grid(axis='y', alpha=.22); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_lens_abundance.png', dpi=150, facecolor='white')
    print('B wrote fig_lens_abundance.png')


# ---------------------------------------------------------------- C
def fig_equivalence(t, md):
    """Their six DE claims against our primary-arm confidence intervals; reads the equivalence table, refits nothing."""
    eq = pd.read_csv(gg2.SHARED / 'branches' / PRIMARY_ARM / 'stage_4' / 'equivalence.tsv',
                     sep='\t')
    ours = eq[(eq.tissue == 'lens') & (eq.method == 'edgeR-QL')
              & (eq.norm == 'CPM')].set_index('entry')
    pc, pmd = gg2.counts(PRIMARY_ARM), gg2.metadata()
    n_lens = sum(pmd.loc[c, 'tissue_name'] == 'lens' for c in pc.columns)
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')

    rows = []
    for _, r in theirs.iterrows():
        if r.miRNA not in ours.index:          # below our filter: not tested, no interval
            rows.append(dict(miRNA=r.miRNA, theirs=r.logFC, ours=np.nan, lo=np.nan,
                             hi=np.nan, tested=False, excluded=False))
            continue
        o = ours.loc[r.miRNA]
        rows.append(dict(miRNA=r.miRNA, theirs=r.logFC, ours=o.log2FC,
                         lo=o.ci_lo, hi=o.ci_hi, tested=True,
                         excluded=bool(r.logFC < o.ci_lo or r.logFC > o.ci_hi)))
    d = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    y = np.arange(len(d))
    ax.axvline(0, color='#999', lw=1)
    for i, r in d.iterrows():
        if not r.tested:
            ax.text(-1.55, i, 'not tested (below our expression filter)', fontsize=9.2,
                    color=C_BAD, va='center', ha='left', style='italic')
        else:
            ax.hlines(i, r.lo, r.hi, color=C_US, lw=3.2, zorder=2)
            ax.scatter(r.ours, i, s=64, c=C_US, zorder=3, edgecolor='w', linewidth=.8)
            ax.annotate('excluded' if r.excluded else 'not excluded',
                        (r.theirs, i), xytext=(0, 13), textcoords='offset points',
                        fontsize=8.8, ha='center',
                        color=C_THEM if r.excluded else '#8a6c2f')
        ax.scatter(r.theirs, i, s=130, marker='*', c=C_THEM, zorder=4,
                   edgecolor='w', linewidth=.6)
    ax.set_yticks(y)
    ax.set_yticklabels([e.replace('hsa-', '') for e in d.miRNA], fontsize=10.5)
    ax.set_xlabel('log2 fold change (exfoliation vs control)', fontsize=10.5)
    ax.set_title('Lens: their claimed effects against our confidence intervals',
                 fontsize=13.5, weight='bold', pad=30)
    ax.text(.5, 1.012, f'{PRIMARY_ARM}  ·  n={n_lens} lens  ·  edgeR-QL, CPM  ·  '
            'bars are our 90% CI  ·  stars are the effect they reported',
            transform=ax.transAxes, ha='center', fontsize=10, color='#555')
    ax.set_ylim(-.6, len(d) - .3)
    ax.grid(axis='x', alpha=.22); ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, .03, 1, 1])

    d.insert(1, 'arm', PRIMARY_ARM)
    d.insert(2, 'engine', 'edgeR-QL')
    d.insert(3, 'norm', 'CPM')
    d.insert(4, 'filter', 'section-2.6 (>=10 reads in >=50% of either disease group)')
    OUT_P.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_P / 'fig_lens_claims_vs_our_CIs.png', dpi=150, facecolor='white')
    d.to_csv(OUT_P / 'lens_claims_vs_our_CIs.tsv', sep='\t', index=False)
    print(f'C wrote fig_lens_claims_vs_our_CIs.png  ({int(d.excluded.sum())}/'
          f'{int(d.tested.sum())} tested claims excluded; '
          f'{int((~d.tested).sum())} not tested)')


def main():
    t, md = lens_data()
    fig_composition(t, md)
    fig_abundance(t, md)
    fig_equivalence(t, md)


if __name__ == '__main__':
    main()
