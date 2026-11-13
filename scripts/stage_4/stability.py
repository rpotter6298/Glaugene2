#!/usr/bin/env python3
"""Stage 4 — candidate fragility: leave-one-out, random-3 drops x200, and (reference-only) drop-top-3 counts."""
import sys, os, warnings
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2, norm as N
from de import filtered
from de_edger import run_edger

warnings.filterwarnings('ignore')
os.environ.setdefault('OMP_NUM_THREADS', '1')
OUT = gg2.PLOTS / 'stage_4'; OUT.mkdir(parents=True, exist_ok=True)
NPROC = 12
NORM = 'CPM'          # primary since the 2026 benchmark; see stage_3/norm.py

# Candidates are read from the current stage4_table.tsv for this arm (raw p<0.05 under the lead
# method), not hardcoded — a hardcoded list from an earlier arm silently tested the wrong set.
# Refits use edgeR-QL, the same model that nominated the candidates.
LEAD = 'RPM->edgeR-QL'


def _candidates():
    t = pd.read_csv(gg2.PLOTS / 'stage_4' / 'stage4_table.tsv', sep='\t')
    t = t[t.raw_sig & (t.method == LEAD)]
    return {tis: sorted(set(g.entry)) for tis, g in t.groupby('tissue')}


CANDIDATES = _candidates()
RANDOM_DRAWS, K_DROP = 200, 3
_C = {}


def prep(tis):
    if tis not in _C:
        md = gg2.metadata(); m = gg2.matrix(gg2.PRIMARY)
        rc = pd.read_csv(gg2.SHARED / 'stage_2' / 'read_counts.tsv',
                         sep='\t', index_col=0)['total_reads']
        mt, s = filtered(m, md, tis)
        meta = md.loc[s, ['sex', 'disease']].copy()
        meta['sex'] = meta['sex'].astype(str)
        _C[tis] = (mt, meta, rc)
    return _C[tis]


def fit(tis, drop):
    mt, meta, rc = prep(tis)
    keep = [c for c in mt.columns if c not in set(drop)]
    sub = mt[keep]
    sub = sub.loc[sub.sum(axis=1) > 0]
    r = run_edger(sub, meta.loc[keep], N.size_factors(NORM, sub, rc))
    return r.set_index('entry').rename(columns={'ql_p': 'pvalue',
                                                'ql_log2FC': 'log2FoldChange'})


def job(args):
    tis, kind, drop, tag = args
    r = fit(tis, drop)
    out = []
    for e in [c for c in CANDIDATES[tis] if c in _C[tis][0].index]:
        if e in r.index and pd.notna(r.loc[e, 'pvalue']):
            out.append(dict(tissue=tis, kind=kind, tag=tag, entry=e,
                            p=float(r.loc[e, 'pvalue']),
                            log2FC=float(r.loc[e, 'log2FoldChange'])))
    return out


def main():
    jobs = []
    for tis in CANDIDATES:
        mt, meta, rc = prep(tis)
        jobs.append((tis, 'full', (), 'full'))
        for s in mt.columns:
            jobs.append((tis, 'LOO', (s,), s))
        for e in CANDIDATES[tis]:
            if e in mt.index:
                top3 = tuple(mt.loc[e].nlargest(3).index)
                jobs.append((tis, 'drop-top3', top3, e))
        rng = np.random.default_rng(11)
        for i in range(RANDOM_DRAWS):
            drop = tuple(rng.choice(mt.columns, K_DROP, replace=False))
            jobs.append((tis, 'random', drop, f'r{i}'))
    print(f'{len(jobs)} refits, {NPROC} workers')
    with Pool(NPROC) as pool:
        res = [x for chunk in pool.map(job, jobs) for x in chunk]
    d = pd.DataFrame(res)
    d.to_csv(OUT / 'stability_raw.tsv', sep='\t', index=False)

    rows = []
    for tis in CANDIDATES:
        full = d[(d.tissue == tis) & (d.kind == 'full')].set_index('entry')
        loo = d[(d.tissue == tis) & (d.kind == 'LOO')]
        dt3 = d[(d.tissue == tis) & (d.kind == 'drop-top3')]
        rnd = d[(d.tissue == tis) & (d.kind == 'random')]
        for e in CANDIDATES[tis]:
            if e not in full.index:
                continue
            l = loo[loo.entry == e]
            t3 = dt3[(dt3.entry == e) & (dt3.tag == e)]
            rr = rnd[rnd.entry == e].p
            rows.append(dict(
                tissue=tis, entry=e,
                p_full=full.loc[e, 'p'], log2FC=full.loc[e, 'log2FC'],
                p_loo_worst=float(l.p.max()) if len(l) else np.nan,
                loo_driver=l.loc[l.p.idxmax(), 'tag'] if len(l) else '',
                p_drop_top3=float(t3.p.iloc[0]) if len(t3) else np.nan,
                rand_frac_sig=float((rr < .05).mean()) if len(rr) else np.nan,
                rand_p10=float(rr.quantile(.10)) if len(rr) else np.nan,
                rand_p50=float(rr.median()) if len(rr) else np.nan,
                rand_p90=float(rr.quantile(.90)) if len(rr) else np.nan,
                stable=bool(len(l) and l.p.max() < .05)))
    S = pd.DataFrame(rows).sort_values(['tissue', 'p_full'])
    S.to_csv(OUT / 'stability.tsv', sep='\t', index=False)
    pd.set_option('display.width', 200)
    print('\n' + S.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nsurvive leave-one-out at p<0.05: {int(S.stable.sum())}/{len(S)}')
    print(f'keep p<0.05 in >=90% of {RANDOM_DRAWS} random {K_DROP}-sample dropouts: '
          f'{int((S.rand_frac_sig >= .9).sum())}/{len(S)}')
    print(f'wrote {OUT}/stability.tsv')


if __name__ == '__main__':
    main()
