#!/usr/bin/env python3
"""Stage 3 — differential expression candidates under several normalisations, uncorrected (Stage 4 corrects)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2, norm as N

warnings.filterwarnings('ignore')

# pydeseq2 is imported inside run() only: `filtered` is imported pipeline-wide, and a
# module-level pydeseq2 import would fail everywhere whenever pydeseq2 is missing.
# Run with /home/rpotter/miniconda3/bin/python3; the system python lacks statsmodels,
# pydeseq2 and nbsr.

OUT = gg2.PLOTS / 'stage_3'; OUT.mkdir(parents=True, exist_ok=True)
ARM = gg2.PRIMARY
TISSUES = ['lens', 'tear']
NORMS = ['TMM', 'median-of-ratios', 'CPM', 'input-RNA']


def filtered(m, md, tissue):
    """Locked Stage 1 rule: >=10 reads in >=50% of a tissue x disease-class group."""
    s = [c for c in m.columns if md.loc[c, 'tissue_name'] == tissue]
    mt = m[s]
    ok = None
    for d in ['control', 'diseased']:
        g = [c for c in s if md.loc[c, 'disease'] == d]
        hit = (mt[g] >= 10).sum(axis=1) >= 0.5 * len(g)
        ok = hit if ok is None else (ok | hit)
    return mt[ok], s


def run(counts, meta, size_factors):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts.T, metadata=meta, design='~sex + disease',
                       quiet=True)
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


def main():
    md = gg2.metadata()
    m = gg2.matrix(ARM)
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
            r = run(mt, meta, sf).reset_index().rename(columns={'index': 'entry'})
            r['tissue'], r['norm'] = tis, nm
            r['kind'] = N.FAMILIES[nm][0]
            rows.append(r)
            sig = int((r.pvalue < 0.05).sum())
            print(f'  {nm:18s} ({N.FAMILIES[nm][0]:8s})  '
                  f'median log2FC {r.log2FoldChange.median():+.3f}   '
                  f'p<0.05: {sig:3d}/{len(r)}   '
                  f'top: {r.loc[r.pvalue.idxmin(), "entry"]} p={r.pvalue.min():.4f}')
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(OUT / 'de_results.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/de_results.tsv  ({len(d):,} rows)')


if __name__ == '__main__':
    main()
