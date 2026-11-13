#!/usr/bin/env python3
"""Stage 4 - what our own study would have reported at 5 vs 5: draw 5 real XFG and 5 real controls, keep true labels, refit."""
import os
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('GG2_ARM', 'BEA-dedup')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2
import norm as N
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from de import filtered
from fig_power_vs_comparator import run_disease, prep

warnings.filterwarnings('ignore')
OUT = gg2.PLOTS / 'stage_4'
NDRAW = int(os.environ.get('GG2_NDRAW', '500'))
# 'real' (default) keeps true labels and cannot be dismissed as "erased the biology";
# 'random' assigns labels at random -> pure false-positive rate. The two agree closely.
MODE = os.environ.get('GG2_LABELS', 'real')
assert MODE in ('random', 'real')
SUFFIX = '' if MODE == 'real' else '_randomlabels'
NPROC, NORM = 12, 'CPM'
# 'edger' (default) is the same test as the reported DE - the draws must use it, or the
# comparison carries an engine difference as well as a sample-size one. 'pydeseq2' is
# retained to show whether the result depends on the engine.
ENGINE = os.environ.get('GG2_ENGINE', 'edger')
assert ENGINE in ('edger', 'pydeseq2')
ESUF = '' if ENGINE == 'edger' else '_pydeseq2'
RSCRIPT = Path(__file__).resolve().parent / 'smallstudy_edger.R'
R_ENV = {**os.environ, 'R_PROFILE_USER': '/dev/null', 'R_LIBS': '/r_packages'}
THEIR_N, THEIR_P = 5, 0.10
C_NULL, C_THEM = '#8a93a0', '#c1443f'


def draw(seed, meta):
    """Sample list for one 5-vs-5 study. Factored out so the edgeR and pydeseq2 paths
    resample identically (same seed, same RNG calls); any difference is the engine."""
    rng = np.random.default_rng(seed)
    if MODE == 'random':
        take = list(rng.choice(list(meta.index), 2 * THEIR_N, replace=False))
        lab = ['diseased'] * THEIR_N + ['control'] * THEIR_N
    else:
        dis = meta.index[meta.disease == 'diseased']
        con = meta.index[meta.disease == 'control']
        take = (list(rng.choice(dis, THEIR_N, replace=False)) +
                list(rng.choice(con, THEIR_N, replace=False)))
        lab = None
    return take, lab


def one(seed):
    """One 5-vs-5 study through pydeseq2. See ENGINE."""
    mt, meta, rc = prep()
    rng = np.random.default_rng(seed)
    if MODE == 'random':
        take = list(rng.choice(list(meta.index), 2 * THEIR_N, replace=False))
        msub = meta.loc[take, ['sex', 'disease']].copy()
        # labels assigned at random, so every hit below is a false positive
        msub['disease'] = ['diseased'] * THEIR_N + ['control'] * THEIR_N
    else:
        dis = meta.index[meta.disease == 'diseased']
        con = meta.index[meta.disease == 'control']
        take = (list(rng.choice(dis, THEIR_N, replace=False)) +
                list(rng.choice(con, THEIR_N, replace=False)))
        msub = meta.loc[take, ['sex', 'disease']].copy()
    sub = mt[take].copy().astype(int)
    try:
        r = run_disease(sub, msub, N.size_factors(NORM, sub, rc))
    except Exception:
        return None
    r = r.dropna(subset=['pvalue'])
    hit = r[r.pvalue < THEIR_P]
    lfc = np.abs(hit.log2FoldChange.values)
    return dict(seed=seed, n_p10=int(len(hit)), n_p05=int((r.pvalue < .05).sum()),
                # hits that clear their p threshold AND are at least as large as the
                # weakest effect they reported - the profile they actually published
                n_match=int((lfc >= 0.99).sum()), lfcs=list(lfc))


def run_edger(seeds, mt, meta):
    """All draws in one R process: 500 Rscript startups would dominate the runtime."""
    import subprocess, tempfile
    rows = []
    for sd in seeds:
        take, lab = draw(sd, meta)
        rows.append([str(sd)] + list(take))
    md = meta[['disease']].copy()
    if MODE == 'random':
        # labels are per-draw, so a single meta table cannot express them; the
        # random-label mode stays on the pydeseq2 path until it is needed here
        raise SystemExit('GG2_LABELS=random is not implemented for GG2_ENGINE=edger')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mt.to_csv(td / 'c.tsv', sep='\t')
        md.to_csv(td / 'm.tsv', sep='\t', index_label='sample')
        pd.DataFrame(rows, columns=['seed'] + [f's{i}' for i in range(2 * THEIR_N)]
                     ).to_csv(td / 'd.tsv', sep='\t', index=False)
        p = subprocess.run(
            ['Rscript', str(RSCRIPT), str(td / 'c.tsv'), str(td / 'm.tsv'),
             str(td / 'd.tsv'), str(td / 'h.tsv'), str(td / 'su.tsv'), str(THEIR_P)],
            capture_output=True, text=True, env=R_ENV)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        print(p.stdout.rstrip(), flush=True)
        summ = pd.read_csv(td / 'su.tsv', sep='\t').dropna(subset=['n_p10'])
        hits = pd.read_csv(td / 'h.tsv', sep='\t')
    lfc = {s: np.abs(g.logFC.values) for s, g in hits.groupby('seed')}
    out = []
    for r in summ.itertuples():
        v = lfc.get(r.seed, np.array([]))
        out.append(dict(seed=int(r.seed), n_p10=int(r.n_p10), n_p05=int(r.n_p05),
                        n_match=int((v >= 0.99).sum()), lfcs=list(v)))
    return out


def main():
    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')
    their_n, their_lfc = len(theirs), theirs.logFC.abs().values
    mt, meta, _ = prep()
    print(f'lens: {mt.shape[0]} miRNAs, {len(meta)} samples  ->  {NDRAW} '
          f'{THEIR_N}-vs-{THEIR_N} studies, labels={MODE}, engine={ENGINE}', flush=True)
    seeds = range(4000, 4000 + NDRAW)
    if ENGINE == 'edger':
        res = run_edger(list(seeds), mt, meta)
    else:
        with Pool(NPROC) as pool:
            res = [r for r in pool.map(one, seeds) if r]
    d = pd.DataFrame(res)
    d[['seed', 'n_p10', 'n_p05', 'n_match']].to_csv(
        OUT / f'smallstudy_5v5{SUFFIX}{ESUF}.tsv', sep='\t', index=False)
    # per-hit effect sizes as well as counts, so figures (S4) redraw without refitting
    pd.DataFrame([(int(r.seed), float(v)) for r in d.itertuples() for v in r.lfcs],
                 columns=['seed', 'abs_log2FC']).to_csv(
        OUT / f'smallstudy_5v5_hits{SUFFIX}{ESUF}.tsv', sep='\t', index=False)
    # "as good as theirs" = their published profile reproduced: >= their hit count,
    # each hit >= their weakest effect, and >=1 at p<0.05
    as_good = float(((d.n_match >= their_n) & (d.n_p05 >= 1)).mean())
    all_lfc = np.concatenate([np.asarray(x) for x in d.lfcs if len(x)])
    ge = float((d.n_p10 >= their_n).mean())

    print(f'  hits at p<{THEIR_P:g}: median {d.n_p10.median():.0f}, '
          f'mean {d.n_p10.mean():.1f}, 90th pct {np.percentile(d.n_p10, 90):.0f}, '
          f'max {d.n_p10.max()}')
    print(f'  {ge:.0%} of random 5-v-5 studies produced >= {their_n} hits')
    print(f'  hits at p<0.05: median {d.n_p05.median():.0f}, '
          f'{float((d.n_p05 >= 1).mean()):.0%} of studies had >= 1')
    print(f'  |log2FC| of those hits: median {np.median(all_lfc):.2f}, '
          f'{float((all_lfc >= their_lfc.min()).mean()):.0%} are >= their smallest '
          f'({their_lfc.min():.2f}), {float((all_lfc >= their_lfc.max()).mean()):.0%} '
          f'>= their largest ({their_lfc.max():.2f})')

    fig, (a, b) = plt.subplots(1, 2, figsize=(13.8, 5.8))

    # A: how many hits
    bins = np.arange(-.5, max(d.n_p10.max(), their_n) + 2)
    a.hist(d.n_p10, bins=bins, color=C_NULL, edgecolor='white', linewidth=.6)
    a.axvline(their_n, color=C_THEM, lw=2.6, zorder=4)
    a.annotate(f'Genes 2022\n{their_n} miRNAs', (their_n, .96),
               xycoords=('data', 'axes fraction'), xytext=(16, 0),
               textcoords='offset points', color=C_THEM, fontsize=10,
               weight='bold', va='top', linespacing=1.5)
    a.set_xlabel(f'miRNAs at p < {THEIR_P:g}', fontsize=10.5)
    a.set_ylabel(f'{THEIR_N}-vs-{THEIR_N} studies (of {len(d):,})', fontsize=10.5)
    a.set_title('A   Hits at p < 0.1', fontsize=12.5, weight='bold', loc='left')
    a.text(.97, .55, f'median {d.n_p10.median():.0f}\n'
           f'{ge:.0%} reached {their_n} or more',
           transform=a.transAxes, ha='right', va='top', fontsize=9.6, color='#333',
           linespacing=1.6, bbox=dict(fc='white', ec='#ddd', alpha=.94))
    a.grid(axis='y', alpha=.22); a.set_axisbelow(True)

    # B: how big are they; cap at the 99.5th percentile so a handful of ~11-log2FC
    # noise hits do not squash the comparison region
    top = float(np.percentile(all_lfc, 99.5))
    b.hist(np.clip(all_lfc, 0, top), bins=np.linspace(0, top, 45),
           color=C_NULL, edgecolor='white', linewidth=.5, density=True)
    b.set_xlim(0, top)
    for i, v in enumerate(sorted(their_lfc)):
        b.axvline(v, color=C_THEM, lw=1.8, zorder=4)
    b.annotate('Genes 2022\n6 miRNAs',
               (their_lfc.max(), .96), xycoords=('data', 'axes fraction'),
               xytext=(14, 0), textcoords='offset points', color=C_THEM,
               fontsize=10, weight='bold', va='top', linespacing=1.5)
    b.set_xlabel('|log2 fold change| of each hit', fontsize=10.5)
    b.set_ylabel('density', fontsize=10.5)
    b.set_title('B   Effect sizes of those hits', fontsize=12.5, weight='bold',
                loc='left')
    b.text(.97, .62, f'{float((all_lfc >= their_lfc.min()).mean()):.0%} reach '
           f'{their_lfc.min():.2f}, their smallest\n'
           f'{float((all_lfc >= their_lfc.max()).mean()):.0%} reach '
           f'{their_lfc.max():.2f}, their largest',
           transform=b.transAxes, ha='right', va='top', fontsize=9.6, color='#333',
           linespacing=1.6, bbox=dict(fc='white', ec='#ddd', alpha=.94))
    b.grid(axis='y', alpha=.22); b.set_axisbelow(True)

    fig.suptitle('5-vs-5 Studies on Randomised Labels' if MODE == 'random'
                 else '5-vs-5 Subsets of Our Own Study',
                 fontsize=16.5, weight='bold', y=.978)
    how = ('disease labels assigned at random' if MODE == 'random'
           else 'real disease labels kept, 5 XFG and 5 controls drawn at random')
    fig.text(.5, .912, f'{gg2.PRIMARY}  ·  lens  ·  {len(d):,} draws of 5-vs-5 from '
             f'our own samples, {how}  ·  '
             f'{"edgeR-QL, the Stage 3 test" if ENGINE == "edger" else "pydeseq2 Wald"}'
             f', disease-only design, full-cohort filter',
             ha='center', va='center', fontsize=9.8, color='#555')
    fig.text(.5, .862, f'{as_good:.0%} of these draws reproduce their published '
             f'result or better', ha='center', va='center', fontsize=13,
             weight='bold', color='#8c2f2b')
    fig.tight_layout(rect=[0, .01, 1, .836])
    f = OUT / f'fig_smallstudy_5v5{SUFFIX}{ESUF}.png'
    fig.savefig(f, dpi=170, facecolor='white')
    print(f'\n  AS GOOD AS THEIRS ({their_n}+ hits at p<{THEIR_P:g} with '
          f'|log2FC| >= {their_lfc.min():.2f}, and >=1 at p<0.05): {as_good:.0%} '
          f'of {len(d):,} draws')
    print(f'wrote {f}')


if __name__ == '__main__':
    main()
