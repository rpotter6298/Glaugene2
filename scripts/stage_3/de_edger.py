#!/usr/bin/env python3
"""Stage 3 — edgeR arm: TMM + LRT (matches Genes 2022) and the QL F-test; same design/filter/norms as de.py."""
import sys, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2, norm as N
from de import filtered, NORMS, TISSUES

OUT = gg2.PLOTS / 'stage_3'; OUT.mkdir(parents=True, exist_ok=True)
RSCRIPT = Path(__file__).resolve().parent / 'edger.R'
# ~/.Rprofile prepends a conda R lib with limma 3.58.1; edgeR 4.10.3 needs >= 3.63.6
# and .Rprofile wins over R_LIBS unless suppressed — without these env vars R
# silently resolves the old limma and edgeR fails to load.
R_ENV = {**os.environ, 'R_PROFILE_USER': '/dev/null', 'R_LIBS': '/r_packages'}


def run_edger(counts, meta, sf):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        counts.to_csv(td / 'c.tsv', sep='\t')
        meta.to_csv(td / 'm.tsv', sep='\t')
        pd.DataFrame({'size_factor': sf.reindex(counts.columns)}).to_csv(
            td / 's.tsv', sep='\t', index_label='sample')
        p = subprocess.run(
            ['Rscript', str(RSCRIPT), str(td / 'c.tsv'), str(td / 'm.tsv'),
             str(td / 's.tsv'), str(td / 'o.tsv')],
            capture_output=True, text=True, env=R_ENV)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        print(p.stdout.rstrip())
        return pd.read_csv(td / 'o.tsv', sep='\t')


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
            print(f'  {nm} ({N.FAMILIES[nm][0]})')
            r = run_edger(mt, meta, N.size_factors(nm, mt, rc, ng))
            r['tissue'], r['norm'], r['kind'] = tis, nm, N.FAMILIES[nm][0]
            rows.append(r)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(OUT / 'de_edger.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/de_edger.tsv  ({len(d):,} rows)')


if __name__ == '__main__':
    main()
