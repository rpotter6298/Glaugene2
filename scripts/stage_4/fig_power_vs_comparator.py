#!/usr/bin/env python3
"""Stage 4 — could Genes 2022 (n = 5 per group, six lens miRNAs at |log2FC| 0.99-2.05) have found what it reports?"""
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-dedup')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import norm as N
import numpy as np
import pandas as pd
from de import filtered
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
OUT = gg2.PLOTS / 'stage_4'
TISSUE, NORM = 'lens', 'CPM'
THEIR_N = 5                      # Genes 2022: 5 cases, 5 controls
CURVE_LFC = [.25, .5, .75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
NSIM_CURVE, NSIM_CLAIM = 150, 200
NPROC = 12
N_SPIKE = 30
C_THEM, C_US = '#c1443f', '#3a6ea5'
_C = {}


def run_disease(counts, meta, size_factors):
    """Stage 3's fit with a disease-only design: 5-per-group subsamples are often
    single-sex, which makes ~ sex + disease rank-deficient (pydeseq2 LinAlgError)."""
    dds = DeseqDataSet(counts=counts.T, metadata=meta, design='~disease', quiet=True)
    dds.fit_size_factors()
    dds.obs['size_factors'] = np.asarray(size_factors.reindex(counts.columns),
                                         dtype=float)
    dds.fit_genewise_dispersions()
    dds.fit_dispersion_trend()
    dds.fit_dispersion_prior()
    dds.fit_MAP_dispersions()
    dds.fit_LFC()
    dds.calculate_cooks()
    dds.refit()
    st = DeseqStats(dds, contrast=['disease', 'diseased', 'control'], quiet=True)
    st.summary()
    return st.results_df


def prep():
    if not _C:
        md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
        rc = pd.read_csv(gg2.SHARED / 'stage_2' / 'read_counts.tsv', sep='\t',
                         index_col=0)['total_reads']
        mt, s = filtered(m, md, TISSUE)
        meta = md.loc[s, ['sex', 'disease']].copy()
        meta['sex'] = meta['sex'].astype(str)
        _C['d'] = (mt, meta, rc)
    return _C['d']


def one(args):
    """Subsample to n per group, inject lfc into `targets`, refit, count detections."""
    n, lfc, seed, targets = args
    mt, meta, rc = prep()
    rng = np.random.default_rng(seed)
    dis = meta.index[meta.disease == 'diseased']
    con = meta.index[meta.disease == 'control']
    if n > min(len(dis), len(con)):
        return None
    take = list(rng.choice(dis, n, replace=False)) + list(rng.choice(con, n, replace=False))
    sub, msub = mt[take].copy().astype(float), meta.loc[take]
    spikes = (list(targets) if targets is not None
              else list(rng.choice(mt.index, N_SPIKE, replace=False)))
    case = msub.index[msub.disease == 'diseased']
    for e in spikes:
        sub.loc[e, case] *= 2.0 ** lfc
    sub = sub.round().astype(int)
    try:
        r = run_disease(sub, msub, N.size_factors(NORM, sub, rc))
    except Exception:
        return None
    r = r.dropna(subset=['pvalue'])
    hit = r.loc[r.index.intersection(spikes), 'pvalue'] < .05
    return dict(n=n, lfc=lfc, detected=int(hit.sum()), tested=int(len(hit)))


def simulate():
    mt, meta, _ = prep()
    our_n = min(int((meta.disease == 'diseased').sum()),
                int((meta.disease == 'control').sum()))
    print(f'lens: {mt.shape[0]} miRNAs, {(meta.disease == "diseased").sum()} XFG / '
          f'{(meta.disease == "control").sum()} control  ->  balanced n = {our_n}',
          flush=True)

    # GG2_REPLOT=1 redraws from the cached tsv instead of refitting ~3,000 models
    cache = OUT / 'power_vs_comparator.tsv'
    if os.environ.get('GG2_REPLOT') and cache.exists():
        prev = pd.read_csv(cache, sep='\t')
        curve = prev[prev.kind == 'curve'][['n', 'lfc', 'power']]
        claims = prev[prev.kind == 'claim'].drop(columns='kind')
        theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')
        missing = [m for m in theirs.miRNA if m.replace('hsa-', '')
                   not in set(claims.mirna)]
        print(f'replotting from {cache}')
        return curve, claims, theirs, missing, our_n

    jobs = [(n, lfc, 7000 + i, None) for n in (THEIR_N, our_n) for lfc in CURVE_LFC
            for i in range(NSIM_CURVE)]
    with Pool(NPROC) as pool:
        res = [r for r in pool.map(one, jobs) if r]
    curve = (pd.DataFrame(res).groupby(['n', 'lfc'])
             .apply(lambda g: g.detected.sum() / g.tested.sum())
             .rename('power').reset_index())
    print('\npower vs effect size (injected into random miRNAs):')
    for n, g in curve.groupby('n'):
        print(f'  n={int(n):2d}: ' + '  '.join(
            f'{2**r.lfc:.1f}x:{r.power:.0%}' for _, r in g.iterrows()), flush=True)

    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')
    present = [r for _, r in theirs.iterrows() if r.miRNA in mt.index]
    missing = [r.miRNA for _, r in theirs.iterrows() if r.miRNA not in mt.index]
    rows = []
    for r in present:
        lfc = abs(r.logFC)
        for n in (THEIR_N, our_n):
            jobs = [(n, lfc, 9000 + i, [r.miRNA]) for i in range(NSIM_CLAIM)]
            with Pool(NPROC) as pool:
                out = [x for x in pool.map(one, jobs) if x]
            d = pd.DataFrame(out)
            rows.append(dict(mirna=r.miRNA.replace('hsa-', ''), logFC=r.logFC,
                             n_per_group=n, power=d.detected.sum() / d.tested.sum(),
                             n_sim=int(d.tested.sum())))
            print(f'  {r.miRNA.replace("hsa-", ""):14s} |log2FC| {lfc:4.2f}  n={n:2d} '
                  f'-> power {rows[-1]["power"]:.0%}', flush=True)
    claims = pd.DataFrame(rows)
    pd.concat([curve.assign(kind='curve'), claims.assign(kind='claim')]).to_csv(
        OUT / 'power_vs_comparator.tsv', sep='\t', index=False)
    return curve, claims, theirs, missing, our_n


def main():
    curve, claims, theirs, missing, our_n = simulate()

    fig, ax = plt.subplots(figsize=(12.6, 6.8))
    FC = [1.2, 1.5, 2, 3, 4, 5]
    for n, col, lab in ((our_n, C_US, f'our study  —  n = {our_n} per group'),
                        (THEIR_N, C_THEM, f'Genes 2022  —  n = {THEIR_N} per group')):
        g = curve[curve.n == n].sort_values('lfc')
        ax.plot(g.lfc, g.power, '-', color=col, lw=2.6, marker='o', ms=4.5,
                label=lab, zorder=4)
    ax.axhline(.8, color='#888', ls=':', lw=1.3, zorder=2)
    ax.text(.99, .805, '80% power', transform=ax.get_yaxis_transform(),
            ha='right', fontsize=9, color='#666', va='bottom')

    cl = claims[claims.n_per_group == THEIR_N].copy()
    cl['lfc_abs'] = cl.logFC.abs()
    cl = cl.sort_values('lfc_abs')
    for i, (_, r) in enumerate(cl.iterrows()):
        ax.plot([r.lfc_abs, r.lfc_abs], [0, r.power], color='#8c2f2b', lw=1,
                ls='--', alpha=.55, zorder=3)
        ax.scatter([r.lfc_abs], [r.power], s=80, marker='X', color='#8c2f2b',
                   edgecolor='white', linewidth=1.1, zorder=6)
        ax.annotate(f'{r.mirna}\n{r.power:.0%}', (r.lfc_abs, r.power),
                    xytext=(0, -14 if i % 2 else 13), textcoords='offset points',
                    ha='center', va='top' if i % 2 else 'bottom', fontsize=8,
                    color='#8c2f2b', weight='bold', linespacing=1.4, zorder=6)
    ax.scatter([], [], s=80, marker='X', color='#8c2f2b', edgecolor='white',
               linewidth=1.1, label='their claims, at their own n = 5')
    ax.text(np.log2(2.45), .30,
            'The crosses sit above the red curve because their five\n'
            'miRNAs are abundant ones. Abundance drives power;\n'
            'the curve is for an average miRNA.',
            fontsize=8.6, color='#8c2f2b', va='center', linespacing=1.6, zorder=6)

    ax.set_xticks([np.log2(t) for t in FC])
    ax.set_xticklabels([f'{t:g}x' for t in FC], fontsize=10.5)
    ax.set_xlim(min(CURVE_LFC) - .05, max(CURVE_LFC) + .12)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel('size of the effect being looked for  (fold change)', fontsize=11)
    ax.set_ylabel('chance of finding it  (p < 0.05)', fontsize=11)
    ax.legend(fontsize=10, loc='lower right', frameon=True, facecolor='white',
              edgecolor='#ddd', framealpha=.95).set_zorder(7)
    ax.grid(alpha=.22); ax.set_axisbelow(True)

    lo, hi = cl.power.min(), cl.power.max()
    npos = int((theirs.p < .05).sum())
    fig.suptitle('Is our negative result just low power?', fontsize=16.5,
                 weight='bold', y=.978)
    fig.text(.5, .916, f'{gg2.PRIMARY}  ·  lens  ·  simulated by resampling our own '
             f'samples and injecting a known effect  ·  same filter, model and test '
             f'as Stage 3', ha='center', va='center', fontsize=10, color='#555')
    note = (f'No. Effects the size they report would be found {lo:.0%}–{hi:.0%} of the '
            f'time even at n = 5 — and we have {our_n} per group.\n'
            f'{len(theirs) - npos} of their {len(theirs)} reported miRNAs had p > 0.05 '
            f'in their own paper; they were selected on fold change, not significance.')
    fig.text(.5, .055, note, ha='center', va='center', fontsize=10.4, color='#222',
             linespacing=1.7)
    fig.tight_layout(rect=[0, .105, 1, .885])
    f = OUT / 'fig_power_vs_comparator.png'
    fig.savefig(f, dpi=170, facecolor='white')
    print(f'\nwrote {f}')


if __name__ == '__main__':
    main()
