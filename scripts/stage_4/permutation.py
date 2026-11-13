#!/usr/bin/env python3
"""Stage 4 — permutation calibration: shuffle disease labels, rerun, report the permutation p-value (never the null mean)."""
import sys, os, warnings
# Must precede the numpy import: BLAS reads these once at load, and 12 workers each spawning a
# thread pool oversubscribes the box enough to cost an order of magnitude in wall time.
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2, norm as N
from de import filtered, run

warnings.filterwarnings('ignore')

OUT = gg2.PLOTS / 'stage_4'; OUT.mkdir(parents=True, exist_ok=True)
NPERM, NPROC = 1000, 12
COMBOS = [('lens', 'CPM'), ('tear', 'CPM'), ('tear', 'total-reads')]
_CACHE = {}


def _prep(tis):
    if tis not in _CACHE:
        md = gg2.metadata(); m = gg2.matrix(gg2.PRIMARY)
        rc = pd.read_csv(gg2.SHARED / 'stage_2' / 'read_counts.tsv',
                         sep='\t', index_col=0)['total_reads']
        mt, s = filtered(m, md, tis)
        meta = md.loc[s, ['sex', 'disease']].copy()
        meta['sex'] = meta['sex'].astype(str)
        _CACHE[tis] = (mt, meta, rc)
    return _CACHE[tis]


def one(job):
    tis, nm, seed = job
    mt, meta, rc = _prep(tis)
    m2 = meta.copy()
    if seed is not None:
        m2['disease'] = np.random.default_rng(seed).permutation(m2['disease'].values)
    sf = N.size_factors(nm, mt, rc)
    r = run(mt, m2, sf).dropna(subset=['pvalue'])
    p = r.pvalue.values
    n = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    top = np.sort(p)[:10]
    return dict(tissue=tis, norm=nm, seed=seed,
                p05=int((p < .05).sum()), p01=int((p < .01).sum()),
                BH10=int((q < .10).sum()), min_p=float(p.min()),
                **{f'p_rank{i+1}': float(v) for i, v in enumerate(top)})


def main():
    jobs = [(t, n, None) for t, n in COMBOS]
    jobs += [(t, n, 7000 + i) for t, n in COMBOS for i in range(NPERM)]
    print(f'{len(jobs)} runs ({len(COMBOS)} real + {len(COMBOS)}x{NPERM} permuted), '
          f'{NPROC} workers')
    with Pool(NPROC) as pool:
        res = pool.map(one, jobs)
    d = pd.DataFrame(res)
    d.to_csv(OUT / 'permutation.tsv', sep='\t', index=False)

    print(f'\n{"tissue":6s} {"norm":17s} {"metric":7s} {"real":>5s} {"null med":>9s} '
          f'{"null p90":>9s} {"perm p":>7s}')
    summ = []
    for (tis, nm), g in d.groupby(['tissue', 'norm']):
        real = g[g.seed.isna()].iloc[0]
        null = g[g.seed.notna()]
        for metric in ['p05', 'p01', 'BH10']:
            nv = null[metric].values; rv = real[metric]
            pp = (nv >= rv).mean()
            print(f'{tis:6s} {nm:17s} {metric:7s} {rv:5.0f} {np.median(nv):9.1f} '
                  f'{np.quantile(nv, .9):9.1f} {pp:7.2f}')
            summ.append(dict(tissue=tis, norm=nm, metric=metric, real=int(rv),
                             null_median=float(np.median(nv)),
                             null_mean=float(nv.mean()),
                             null_p90=float(np.quantile(nv, .9)), perm_p=float(pp)))
    pd.DataFrame(summ).to_csv(OUT / 'permutation_summary.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/permutation.tsv and permutation_summary.tsv')


if __name__ == '__main__':
    main()
