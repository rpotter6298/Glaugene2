#!/usr/bin/env python3
"""Stage 3 — transform sweep: voom and Box-Cox with limma empirical Bayes (same design/filter/norms as de.py)."""
import sys, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2, norm as N
from de import filtered, NORMS, TISSUES

OUT = gg2.PLOTS / 'stage_3'; OUT.mkdir(parents=True, exist_ok=True)
RSCRIPT = Path(__file__).resolve().parent / 'limma_arm.R'
# ~/.Rprofile prepends a conda R library with limma 3.58.1; suppress it (see de_edger.py)
R_ENV = {**os.environ, 'R_PROFILE_USER': '/dev/null', 'R_LIBS': '/r_packages'}


def _call(mode, mat, meta, sf):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        mat.to_csv(td / 'x.tsv', sep='\t')
        meta.to_csv(td / 'm.tsv', sep='\t')
        pd.DataFrame({'size_factor': sf.reindex(mat.columns)}).to_csv(
            td / 's.tsv', sep='\t', index_label='sample')
        p = subprocess.run(['Rscript', str(RSCRIPT), mode, str(td / 'x.tsv'),
                            str(td / 'm.tsv'), str(td / 's.tsv'), str(td / 'o.tsv')],
                           capture_output=True, text=True, env=R_ENV)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        print(p.stdout.rstrip())
        return pd.read_csv(td / 'o.tsv', sep='\t')


def boxcox_matrix(counts, sf):
    """Size-factor normalise, then one global Box-Cox lambda over all values."""
    norm_ = counts.div(sf.reindex(counts.columns), axis=1)
    off = norm_[norm_ > 0].min().min() / 2          # offset for the zeros
    v = (norm_ + off).values.ravel()
    lam = stats.boxcox_normmax(v, method='mle')
    t = stats.boxcox(v, lmbda=lam).reshape(norm_.shape)
    return pd.DataFrame(t, index=norm_.index, columns=norm_.columns), lam


def main():
    md = gg2.metadata(); m = gg2.matrix(gg2.PRIMARY)
    rc = pd.read_csv(gg2.SHARED / 'stage_2' / 'read_counts.tsv',
                     sep='\t', index_col=0)['total_reads']
    ng = gg2.rna_input()
    rows = []
    for tis in TISSUES:
        mt, s = filtered(m, md, tis)
        meta = md.loc[s, ['sex', 'disease']].copy()
        meta['sex'] = meta['sex'].astype(str)
        print(f'\n=== {tis}: {mt.shape[0]} entries, {len(s)} samples ===')
        for nm in NORMS:
            sf = N.size_factors(nm, mt, rc, ng)
            print(f'  {nm} ({N.FAMILIES[nm][0]})')
            r = _call('voom', mt, meta, sf)
            r['method'] = 'voom'
            rows.append(r.assign(tissue=tis, **{'norm': nm}))
            bc, lam = boxcox_matrix(mt, sf)
            print(f'    box-cox lambda = {lam:.3f}')
            r2 = _call('matrix', bc, meta, sf)
            r2['method'] = 'box-cox'
            rows.append(r2.assign(tissue=tis, **{'norm': nm, 'lambda': lam}))
    d = pd.concat(rows, ignore_index=True)
    d['kind'] = d['norm'].map(lambda x: N.FAMILIES[x][0])
    d.to_csv(OUT / 'de_transform.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/de_transform.tsv  ({len(d):,} rows)')


if __name__ == '__main__':
    main()
