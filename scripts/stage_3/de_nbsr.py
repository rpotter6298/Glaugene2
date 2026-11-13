#!/usr/bin/env python3
"""Stage 3 — NBSR: negative-binomial softmax regression (Jun, Halushka & McCall, Biostatistics 2026; https://github.com/junseonghwan/nbsr)."""
import sys, shutil, warnings
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2
from de import filtered, TISSUES

warnings.filterwarnings('ignore')
from nbsr.nbsr_config import NBSRConfig
from nbsr.main import run, generate_results

OUT = gg2.PLOTS / 'stage_3'; OUT.mkdir(parents=True, exist_ok=True)
WORK = gg2.PLOTS / 'stage_3' / '.nbsr'
ITERATIONS = 5000
COVARIATES = ['sex', 'disease']          # same design as every other arm


def prepare(tis, md, m, rc, work):
    """NBSR wants Y.csv (features x samples) and X.csv (sample metadata)."""
    mt, s = filtered(m, md, tis)
    work.mkdir(parents=True, exist_ok=True)
    mt.to_csv(work / 'Y.csv', index=True)
    X = pd.DataFrame({
        'sample': s,
        'sex': md.loc[s, 'sex'].astype(str).values,
        'disease': md.loc[s, 'disease'].values,
        'lib_size': rc.reindex(s).values.astype(float),     # total sequenced reads
        'miRNA_reads': mt[s].sum().values.astype(float),    # reads assigned to miRNA
    })
    X.to_csv(work / 'X.csv', index=False)
    return mt, s


def main():
    md = gg2.metadata(); m = gg2.matrix(gg2.PRIMARY)
    rc = pd.read_csv(gg2.SHARED / 'stage_2' / 'read_counts.tsv',
                     sep='\t', index_col=0)['total_reads']
    rows = []
    for tis in TISSUES:
        work = WORK / tis
        if work.exists():
            shutil.rmtree(work)
        mt, s = prepare(tis, md, m, rc, work)
        print(f'\n=== {tis}: {mt.shape[0]} features x {len(s)} samples ===', flush=True)
        cfg = NBSRConfig(work / 'Y.csv', work / 'X.csv', work / 'output',
                         COVARIATES, iterations=ITERATIONS,
                         trended_dispersion=False, use_cuda_if_available=False)
        loss, model = run(cfg)
        print(f'  fitted, final loss {float(np.asarray(loss)[-1]):.2f}', flush=True)
        # With ~sex+disease generate_results writes HDF5 only (no frame), and the
        # per-sample columns are identical across sex (no interaction term) — take any.
        h5 = (work / 'output' / 'disease__diseased_vs_control' / 'nbsr_results.h5')
        for absolute in (False, True):
            generate_results(work / 'output', 'disease', 'diseased', 'control',
                             absolute_fc=absolute, recompute_hessian=False)
            with pd.HDFStore(h5, 'r') as store:
                lr, se = store['logRR'], store['se']
                pv, pa = store['pvalue'], store['padj']
            r = pd.DataFrame({
                'entry': lr.index,
                'log2FC': np.log2(np.e) * lr.iloc[:, 0].values,   # NBSR reports natural log
                'lfcSE': np.log2(np.e) * se.iloc[:, 0].values,
                'p': pv.iloc[:, 0].values,
                'padj_nbsr': pa.iloc[:, 0].values,
            })
            r['tissue'] = tis
            r['kind'] = 'absolute' if absolute else 'relative'
            rows.append(r)
            print(f'  {"absolute" if absolute else "relative":8s} '
                  f'median log2FC {r.log2FC.median():+.3f}   '
                  f'p<0.05 {int((r.p < .05).sum())}/{len(r)}   '
                  f'top {r.loc[r.p.idxmin(), "entry"]} p={r.p.min():.4f}', flush=True)
    d = pd.concat(rows, ignore_index=True)
    d.to_csv(OUT / 'de_nbsr.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/de_nbsr.tsv  ({len(d):,} rows)')
    print(f'columns: {list(d.columns)}')


if __name__ == '__main__':
    main()
