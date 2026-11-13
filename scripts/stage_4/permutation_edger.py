#!/usr/bin/env python3
"""Stage 4 — permutation null for the edgeR-QL arm: same shuffles as permutation.py, per-miRNA p-values kept."""
import sys, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2, norm as N
from de import filtered, TISSUES

OUT = gg2.PLOTS / 'stage_4'; OUT.mkdir(parents=True, exist_ok=True)
RSCRIPT = Path(__file__).resolve().parent / 'permutation_edger.R'
# ~/.Rprofile pins an old limma that edgeR 4.10.3 refuses (see de_edger.py)
R_ENV = {**os.environ, 'R_PROFILE_USER': '/dev/null', 'R_LIBS': '/r_packages'}
NPERM = int(os.environ.get('GG2_NPERM', '1000'))
NORM = 'CPM'


def permute(tis):
    md = gg2.metadata(); m = gg2.matrix(gg2.PRIMARY)
    rc = pd.read_csv(gg2.SHARED / 'stage_2' / 'read_counts.tsv',
                     sep='\t', index_col=0)['total_reads']
    mt, s = filtered(m, md, tis)
    meta = md.loc[s, ['sex', 'disease']].copy()
    meta['sex'] = meta['sex'].astype(str)
    sf = N.size_factors(NORM, mt, rc)
    # same seeds as permutation.py: any Wald-vs-QL difference is then the model, not the draw
    lab = meta['disease'].values
    perms = pd.DataFrame(
        [np.random.default_rng(7000 + i).permutation(lab) for i in range(NPERM)],
        columns=meta.index, index=[f'perm{i}' for i in range(NPERM)])
    print(f'=== {tis}: {mt.shape[0]} miRNAs, {len(s)} samples, {NPERM} shuffles ===')

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mt.to_csv(td / 'c.tsv', sep='\t')
        meta.to_csv(td / 'm.tsv', sep='\t')
        pd.DataFrame({'size_factor': sf.reindex(mt.columns)}).to_csv(
            td / 's.tsv', sep='\t', index_label='sample')
        perms.to_csv(td / 'p.tsv', sep='\t', index_label='perm')
        r = subprocess.run(['Rscript', str(RSCRIPT), str(td / 'c.tsv'),
                            str(td / 'm.tsv'), str(td / 's.tsv'), str(td / 'p.tsv'),
                            str(td / 'o.tsv')],
                           capture_output=True, text=True, env=R_ENV)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
        print(r.stdout.rstrip())
        P = pd.read_csv(td / 'o.tsv', sep='\t', index_col=0)
    f = OUT / f'perm_edger_{tis}.tsv.gz'
    P.to_csv(f, sep='\t')
    print(f'wrote {f}')
    return P


def main():
    rows = []
    for tis in TISSUES:
        P = permute(tis)
        real, null = P.iloc[0].values, P.iloc[1:].values
        n, nperm = P.shape[1], null.shape[0]
        obs = int((real < .05).sum())
        hits = (null < .05).sum(axis=1)
        rows.append(dict(
            tissue=tis, n_mirnas=n, n_perm=nperm, real_p05=obs,
            theory_p05=.05 * n,
            null_median=float(np.median(hits)), null_mean=float(hits.mean()),
            null_sd=float(hits.std()),
            null_q25=float(np.percentile(hits, 25)),
            null_q75=float(np.percentile(hits, 75)),
            null_q90=float(np.percentile(hits, 90)),
            binomial_sd=float(np.sqrt(n * .05 * .95)),
            perm_p=float(((hits >= obs).sum() + 1) / (nperm + 1))))  # +1: a permutation p is never 0
        print(f'  {tis}: real {obs}, theory {.05*n:.1f}, null median '
              f'{np.median(hits):.0f} (mean {hits.mean():.1f}, sd {hits.std():.1f} '
              f'vs binomial {np.sqrt(n*.05*.95):.1f}), perm p = {rows[-1]["perm_p"]:.3f}')
    pd.DataFrame(rows).to_csv(OUT / 'perm_edger_summary.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/perm_edger_summary.tsv')


if __name__ == '__main__':
    main()
