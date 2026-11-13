#!/usr/bin/env python3
"""Stage 4 — multiple-testing correction (BH + Storey q) across every Stage 3 run."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import UnivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

S3 = gg2.PLOTS / 'stage_3'
OUT = gg2.PLOTS / 'stage_4'; OUT.mkdir(parents=True, exist_ok=True)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


def storey_pi0(p, lam=np.arange(0.05, 0.96, 0.05)):
    p = np.asarray(p, float)
    pi = np.array([(p > l).sum() / (len(p) * (1 - l)) for l in lam])
    try:
        v = float(UnivariateSpline(lam, pi, k=3, s=len(lam) * np.var(pi) * 0.5)(lam[-1]))
    except Exception:
        v = float(np.median(pi))
    return float(np.clip(v, 0, 1))


def storey_q(p):
    pi0 = storey_pi0(p)
    return pi0 * bh(p), pi0


def gather():
    rows = []
    d = pd.read_csv(S3 / 'de_results.tsv', sep='\t')
    d = d.rename(columns={'log2FoldChange': 'log2FC', 'pvalue': 'p'})
    d['method'] = 'pydeseq2'
    rows.append(d[['entry', 'tissue', 'norm', 'kind', 'method', 'log2FC', 'p', 'baseMean']])
    e = pd.read_csv(S3 / 'de_edger.tsv', sep='\t')
    for m, fc, pv in [('edgeR-LRT', 'lrt_log2FC', 'lrt_p'), ('edgeR-QL', 'ql_log2FC', 'ql_p')]:
        x = e[['entry', 'tissue', 'norm', 'kind', fc, pv]].copy()
        x.columns = ['entry', 'tissue', 'norm', 'kind', 'log2FC', 'p']
        x['method'] = m; x['baseMean'] = np.nan
        rows.append(x)
    # NBSR takes each sample's miRNA total as the in-model offset; RPM is internal, not a norm choice
    n = pd.read_csv(S3 / 'de_nbsr.tsv', sep='\t')
    x = n[['entry', 'tissue', 'kind', 'log2FC', 'p']].copy()
    x['norm'] = 'internal-RPM'; x['method'] = 'NBSR'; x['baseMean'] = np.nan
    rows.append(x[['entry', 'tissue', 'norm', 'kind', 'method', 'log2FC', 'p', 'baseMean']])
    t = pd.read_csv(S3 / 'de_transform.tsv', sep='\t')
    x = t[['entry', 'tissue', 'norm', 'kind', 'method', 'log2FC', 'p']].copy()
    x['baseMean'] = np.nan
    rows.append(x)
    return pd.concat(rows, ignore_index=True).dropna(subset=['p'])


def main():
    d = gather()
    out, summ = [], []
    for (tis, nm, knd, meth), g in d.groupby(['tissue', 'norm', 'kind', 'method']):
        g = g.copy()
        g['BH'] = bh(g.p.values)
        g['q'], pi0 = storey_q(g.p.values)
        out.append(g)
        summ.append(dict(tissue=tis, norm=nm, kind=g.kind.iloc[0], method=meth,
                         tested=len(g), pi0=round(pi0, 3),
                         p05=int((g.p < .05).sum()),
                         BH10=int((g.BH < .10).sum()), BH05=int((g.BH < .05).sum()),
                         q10=int((g.q < .10).sum()),
                         min_p=float(g.p.min()), min_BH=float(g.BH.min())))
    D = pd.concat(out, ignore_index=True)
    D.to_csv(OUT / 'corrected.tsv', sep='\t', index=False)
    S = pd.DataFrame(summ).sort_values(['tissue', 'kind', 'norm', 'method'])
    S.to_csv(OUT / 'correction_summary.tsv', sep='\t', index=False)
    print(S.to_string(index=False))
    print(f'\nanything at BH<0.10 anywhere: {int(S.BH10.sum())}')
    print(f'anything at Storey q<0.10   : {int(S.q10.sum())}')
    print(f'\nwrote {OUT}/corrected.tsv and correction_summary.tsv')


if __name__ == '__main__':
    main()
