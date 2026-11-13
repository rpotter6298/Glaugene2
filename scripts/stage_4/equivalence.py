#!/usr/bin/env python3
"""Stage 4 — equivalence testing (TOST, Lakens 2017): per-miRNA states, achieved bound, 90% CI inside ±Δ."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2
from correction import bh

OUT = gg2.PLOTS / 'stage_4'
S3 = gg2.PLOTS / 'stage_3'
DELTAS = [1.0, 1.66]
ALPHA = 0.05
Z = stats.norm.ppf(1 - ALPHA)          # 1.645


def gather():
    """Stage 3 results with an SE: (entry, tissue, norm, method, log2FC, se, p)."""
    rows = []
    f = S3 / 'de_results.tsv'
    if f.exists():
        d = pd.read_csv(f, sep='\t')
        d = d[d.kind == 'relative'].rename(
            columns={'log2FoldChange': 'log2FC', 'lfcSE': 'se', 'pvalue': 'p'})
        d['method'] = 'pydeseq2'
        rows.append(d[['entry', 'tissue', 'norm', 'method', 'log2FC', 'se', 'p']])
    f = S3 / 'de_nbsr.tsv'
    if f.exists():
        d = pd.read_csv(f, sep='\t')
        d = d[d.kind == 'relative'].rename(columns={'lfcSE': 'se'})
        d['method'] = 'NBSR'; d['norm'] = 'internal-RPM'
        rows.append(d[['entry', 'tissue', 'norm', 'method', 'log2FC', 'se', 'p']])
    f = S3 / 'de_edger.tsv'
    if f.exists():
        d = pd.read_csv(f, sep='\t')
        d = d[d.kind == 'relative'].copy()
        d['se'] = np.where(d.ql_F > 0, d.ql_log2FC.abs() / np.sqrt(d.ql_F), np.nan)  # F = t^2 (1 df)
        d = d.rename(columns={'ql_log2FC': 'log2FC', 'ql_p': 'p', 'ql_df': 'df'})
        d['method'] = 'edgeR-QL'
        rows.append(d[['entry', 'tissue', 'norm', 'method', 'log2FC', 'se', 'p', 'df']])
    if not rows:
        raise SystemExit(f'no Stage 3 output with standard errors in {S3}')
    d = pd.concat(rows, ignore_index=True)
    if 'df' not in d:
        d['df'] = np.nan
    return d[np.isfinite(d.se) & (d.se > 0) & np.isfinite(d.log2FC)].copy()


def tost(b, se, delta, df=None):
    """TOST p = max of the two one-sided tests; t where df given (edgeR-QL), z otherwise."""
    lo, hi = (b + delta) / se, (b - delta) / se
    if df is None:
        return np.maximum(stats.norm.sf(lo), stats.norm.cdf(hi))
    df = np.asarray(df, dtype=float)
    use_t = np.isfinite(df)
    safe = np.where(use_t, df, 1.0)
    p_lo = np.where(use_t, stats.t.sf(lo, safe), stats.norm.sf(lo))
    p_hi = np.where(use_t, stats.t.cdf(hi, safe), stats.norm.cdf(hi))
    return np.maximum(p_lo, p_hi)


def crit(df):
    """One-sided critical value: t with the fit's df where given, else z."""
    df = np.asarray(df, dtype=float)
    use_t = np.isfinite(df)
    return np.where(use_t, stats.t.ppf(1 - ALPHA, np.where(use_t, df, 1.0)), Z)


def main():
    d = gather()
    d['crit'] = crit(d.df.values)
    d['ci_lo'] = d.log2FC - d.crit * d.se
    d['ci_hi'] = d.log2FC + d.crit * d.se
    d['achieved_bound'] = d.log2FC.abs() + d.crit * d.se

    for delta in DELTAS:
        tag = f'{delta:g}'.replace('.', 'p')
        d[f'p_tost_{tag}'] = tost(d.log2FC.values, d.se.values, delta, d.df.values)
        d[f'equiv_{tag}'] = d[f'p_tost_{tag}'] < ALPHA
        state = np.where(d[f'equiv_{tag}'] & (d.p >= .05), 'equivalent',
                 np.where(d[f'equiv_{tag}'] & (d.p < .05), 'trivial',
                  np.where(~d[f'equiv_{tag}'] & (d.p < .05), 'different',
                           'inconclusive')))
        d[f'state_{tag}'] = state

    # BH on the TOST p's as well: false claims of equivalence are a multiplicity problem too
    for delta in DELTAS:
        tag = f'{delta:g}'.replace('.', 'p')
        d[f'bh_tost_{tag}'] = np.nan
        for k, g in d.groupby(['tissue', 'norm', 'method']):
            d.loc[g.index, f'bh_tost_{tag}'] = bh(g[f'p_tost_{tag}'].values)

    d.to_csv(OUT / 'equivalence.tsv', sep='\t', index=False)

    summ = []
    print(f'\narm: {gg2.PRIMARY}   alpha={ALPHA}  (90% CI inside +/-delta)\n')
    for (tis, nm, meth), g in d.groupby(['tissue', 'norm', 'method']):
        row = dict(arm=gg2.PRIMARY, tissue=tis, norm=nm, method=meth, n=len(g),
                   median_bound=round(float(g.achieved_bound.median()), 3),
                   q90_bound=round(float(g.achieved_bound.quantile(.90)), 3))
        for delta in DELTAS:
            tag = f'{delta:g}'.replace('.', 'p')
            vc = g[f'state_{tag}'].value_counts()
            row[f'equiv_{tag}'] = int(vc.get('equivalent', 0) + vc.get('trivial', 0))
            row[f'diff_{tag}'] = int(vc.get('different', 0))
            row[f'incon_{tag}'] = int(vc.get('inconclusive', 0))
            row[f'equiv_bh_{tag}'] = int((g[f'bh_tost_{tag}'] < ALPHA).sum())
        summ.append(row)
    S = pd.DataFrame(summ).sort_values(['tissue', 'method', 'norm'])
    S.to_csv(OUT / 'equivalence_summary.tsv', sep='\t', index=False)

    show = ['tissue', 'method', 'norm', 'n', 'equiv_1', 'incon_1', 'diff_1',
            'equiv_bh_1', 'equiv_1p66', 'median_bound', 'q90_bound']
    print(S[show].to_string(index=False))
    print('\nequiv_1     = |log2FC| shown < 1.0 (pre-specified)')
    print('equiv_1p66  = |log2FC| shown < 1.66 (weakest Genes 2022 claim)')
    print('median_bound= median smallest Δ we could exclude, per miRNA')

    prim = d[(d.method == 'edgeR-QL') & (d['norm'] == 'CPM')]
    if len(prim):
        print('\n--- primary cell (edgeR-QL / CPM) ---')
        for tis, g in prim.groupby('tissue'):
            n_eq = int((g.p_tost_1 < ALPHA).sum())
            print(f'  {tis:5s} {n_eq}/{len(g)} miRNAs ({100*n_eq/len(g):.0f}%) shown '
                  f'equivalent at |log2FC|<1.0; median achieved bound '
                  f'{g.achieved_bound.median():.2f}')
    print(f'\nwrote {OUT}/equivalence.tsv and equivalence_summary.tsv')


if __name__ == '__main__':
    main()
