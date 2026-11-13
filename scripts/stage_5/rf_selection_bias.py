#!/usr/bin/env python3
"""Stage 5 — how much AUC does feature-selection leakage manufacture?"""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-raw')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import de
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

OUT = gg2.PLOTS / 'stage_5'
OUT.mkdir(parents=True, exist_ok=True)
K, N_TREES, N_PERM, NPROC = 20, 300, 100, 22
TISSUES = ['lens', 'tear']


def loocv_auc(X, y, leaked, seed=0):
    p = np.empty(len(y))
    if leaked:                                   # selection sees every label, once
        _, pv = stats.ttest_ind(X[y == 1], X[y == 0], axis=0)
        fixed = np.argsort(pv)[:K]
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        if leaked:
            idx = fixed
        else:                                    # re-selected without the held-out sample
            _, pv = stats.ttest_ind(X[tr][y[tr] == 1], X[tr][y[tr] == 0], axis=0)
            idx = np.argsort(pv)[:K]
        clf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                     class_weight='balanced', n_jobs=1)
        clf.fit(X[tr][:, idx], y[tr])
        p[i] = clf.predict_proba(X[i:i + 1, idx])[0, 1]
    return roc_auc_score(y, p)


def main():
    md = gg2.metadata()
    m = gg2.matrix(gg2.PRIMARY)
    rows = []
    for tis in TISSUES:
        sub, s = de.filtered(m, md, tis)
        X = np.log2((sub / sub.sum() * 1e6).T.values + 1)
        y = (md.loc[s, 'disease'] == 'diseased').astype(int).values
        print(f'\n=== {tis}  n={len(y)}  {X.shape[1]} features, top-{K} selected ===', flush=True)
        rng = np.random.default_rng(0)
        perms = [rng.permutation(y) for _ in range(N_PERM)]

        for leaked in (False, True):
            tag = 'leaked' if leaked else 'honest'
            auc = loocv_auc(X, y, leaked)
            null = np.array(Parallel(n_jobs=NPROC)(
                delayed(loocv_auc)(X, yy, leaked, 100 + k)
                for k, yy in enumerate(perms)))
            pv = (1 + (null >= auc).sum()) / (1 + len(null))
            np.save(OUT / f'null_sel_{tag}_{tis}.npy', null)
            print(f'  {tag:7s} AUC {auc:.3f}   null median {np.median(null):.3f}  '
                  f'p90 {np.percentile(null, 90):.3f}   perm p = {pv:.3f}')
            rows.append(dict(tissue=tis, selection=tag, k=K, n=len(y),
                             auc=round(auc, 4),
                             null_median=round(float(np.median(null)), 4),
                             null_p90=round(float(np.percentile(null, 90)), 4),
                             perm_p=round(float(pv), 4), n_perm=N_PERM))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'rf_selection_bias.tsv', sep='\t', index=False)
    print('\n' + d.to_string(index=False))
    for tis in TISSUES:
        a = d[(d.tissue == tis) & (d.selection == 'honest')].auc.iloc[0]
        b = d[(d.tissue == tis) & (d.selection == 'leaked')].auc.iloc[0]
        print(f'  {tis}: leakage inflates AUC by {b - a:+.3f}', flush=True)


if __name__ == '__main__':
    main()
