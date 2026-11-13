#!/usr/bin/env python3
"""Replication A, Table 5 — the most stable tear miRNAs (Jones et al. 2026)."""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'CBGM-v22')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import norm as N
import numpy as np
import pandas as pd

OUT = gg2.PLOTS / 'replication'
OUT.mkdir(parents=True, exist_ok=True)
DETECT_FRAC = 0.90


def genorm_m(log2x):
    """geNorm M: mean SD across samples of log-ratios against every other gene."""
    a = log2x.values
    n = a.shape[0]
    # pairwise log-ratio SDs without an n^2 python loop
    M = np.empty(n)
    for i in range(n):
        d = a[i][None, :] - a          # (n, samples)
        sd = d.std(axis=1, ddof=1)
        M[i] = (sd.sum() - sd[i]) / (n - 1)
    return pd.Series(M, index=log2x.index)


def normfinder_like(log2x):
    """Single-group NormFinder: residual SD after removing gene and sample effects."""
    a = log2x.values
    r = a - a.mean(axis=1, keepdims=True)      # gene effect
    r = r - r.mean(axis=0, keepdims=True)      # sample effect
    return pd.Series(r.std(axis=1, ddof=1), index=log2x.index)


def iqr_scale(s):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    return (s - s.median()) / (q3 - q1)


def main():
    md = gg2.metadata()
    m = gg2.counts(gg2.PRIMARY)
    tear = [c for c in m.columns if md.loc[c, 'tissue_name'] == 'tear']
    t = m[tear]
    n = len(tear)

    keep = t.index[(t > 0).sum(axis=1) >= DETECT_FRAC * n]
    print(f'{len(tear)} tear samples; {len(keep)} miRNAs detected in >={DETECT_FRAC:.0%} '
          f'of them (theirs: 32 samples)')

    sf = N.size_factors('median-of-ratios', t)
    normed = t.loc[keep] / sf.reindex(tear).values
    log2x = np.log2(normed + 1)

    cv = (normed.std(axis=1, ddof=1) / normed.mean(axis=1))
    gn = genorm_m(log2x)
    nf = normfinder_like(log2x)

    d = pd.DataFrame({'CV': cv, 'geNorm': gn, 'NormFinder': nf})
    for c in ['CV', 'geNorm', 'NormFinder']:
        d['scaled_' + c] = iqr_scale(d[c])
    d['total_stability'] = d[['scaled_CV', 'scaled_geNorm',
                              'scaled_NormFinder']].mean(axis=1)
    d = d.sort_values('total_stability')
    d.insert(0, 'rank', range(1, len(d) + 1))
    d.to_csv(OUT / 'tear_stability.tsv', sep='\t')

    theirs = pd.read_csv(gg2.ROOT / 'refs' / 'jones2026_tear_stable10.tsv', sep='\t')
    their10 = list(theirs.miRNA)
    our_rank = {e: r for e, r in zip(d.index, d['rank'])}

    print('\n=== our top 10 most stable ===')
    print(f'{"":3s}{"miRNA":20s}{"CV":>8}{"geNorm":>9}{"NormFinder":>12}'
          f'{"total":>9}   in their top 10?')
    for e, r in d.head(10).iterrows():
        print(f'{int(r["rank"]):<3d}{e.replace("hsa-",""):20s}{r.scaled_CV:>8.3f}'
              f'{r.scaled_geNorm:>9.3f}{r.scaled_NormFinder:>12.3f}'
              f'{r.total_stability:>9.3f}   {"YES" if e in their10 else ""}')

    print('\n=== their top 10, and where each ranks for us ===')
    hit = 0
    for _, r in theirs.iterrows():
        rk = our_rank.get(r.miRNA)
        if rk is None:
            note = 'not detected in >=90% of our samples'
        else:
            note = f'our rank {rk}/{len(d)}'
            if rk <= 10:
                hit += 1
        print(f'  {int(r["rank"]):>2}. {r.miRNA.replace("hsa-",""):18s} {note}')
    print(f'\noverlap in top 10: {hit}/10')


if __name__ == '__main__':
    main()
