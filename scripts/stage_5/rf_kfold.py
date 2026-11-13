#!/usr/bin/env python3
"""Stage 5 — the same classification question under repeated stratified 5-fold CV."""
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
from sklearn.model_selection import StratifiedKFold

OUT = gg2.PLOTS / 'stage_5'
OUT.mkdir(parents=True, exist_ok=True)
K, N_TREES, N_SPLITS, N_REPEAT, N_PERM, NPROC = 20, 300, 5, 1, 100, 22
TISSUES = ['lens', 'tear']
VARIANTS = ['all-features', 'honest', 'leaked']


def kfold_auc(X, y, variant, seed=0, return_oof=False):
    """Mean AUC over N_REPEAT stratified 5-fold splits; with return_oof also the
    pooled out-of-fold probabilities the ROC is drawn from."""
    if variant == 'leaked':
        _, pv = stats.ttest_ind(X[y == 1], X[y == 0], axis=0)
        fixed = np.argsort(pv)[:K]
    aucs = []
    oof = np.zeros(len(y))
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed + rep)
        p = np.empty(len(y))
        for tr_i, te_i in skf.split(X, y):
            if variant == 'all-features':
                idx = np.arange(X.shape[1])
            elif variant == 'leaked':
                idx = fixed
            else:
                _, pv = stats.ttest_ind(X[tr_i][y[tr_i] == 1],
                                        X[tr_i][y[tr_i] == 0], axis=0)
                idx = np.argsort(pv)[:K]
            clf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                         class_weight='balanced', n_jobs=1)
            clf.fit(X[tr_i][:, idx], y[tr_i])
            p[te_i] = clf.predict_proba(X[te_i][:, idx])[:, 1]
        aucs.append(roc_auc_score(y, p))
        oof += p
    if return_oof:
        return float(np.mean(aucs)), float(np.std(aucs)), oof / N_REPEAT
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    md = gg2.metadata()
    m = gg2.matrix(gg2.PRIMARY)
    rows = []
    for tis in TISSUES:
        sub, s = de.filtered(m, md, tis)
        X = np.log2((sub / sub.sum() * 1e6).T.values + 1)
        y = (md.loc[s, 'disease'] == 'diseased').astype(int).values
        print(f'\n=== {tis}  n={len(y)}  {X.shape[1]} features  '
              f'{N_SPLITS}-fold x {N_REPEAT} repeats ===', flush=True)
        rng = np.random.default_rng(0)
        perms = [rng.permutation(y) for _ in range(N_PERM)]

        for v in VARIANTS:
            auc, sd, oof = kfold_auc(X, y, v, return_oof=True)
            pd.DataFrame(dict(sample=s, tissue=tis, variant=v,
                              truth=np.where(y == 1, 'XFG', 'control'),
                              prob_XFG=oof)).to_csv(
                OUT / f'rf_kfold_oof_{v}_{tis}.tsv', sep='\t', index=False)
            null = np.array(Parallel(n_jobs=NPROC)(
                delayed(lambda yy, sd_: kfold_auc(X, yy, v, sd_)[0])(yy, 1000 + k)
                for k, yy in enumerate(perms)))
            pv = (1 + (null >= auc).sum()) / (1 + len(null))
            np.save(OUT / f'null_kfold_{v}_{tis}.npy', null)
            print(f'  {v:13s} AUC {auc:.3f} (sd {sd:.3f} across repeats)   '
                  f'null median {np.median(null):.3f}   perm p = {pv:.3f}',
                  flush=True)
            rows.append(dict(tissue=tis, variant=v, cv=f'{N_SPLITS}-fold',
                             n=len(y), auc=round(auc, 4), auc_sd=round(sd, 4),
                             null_median=round(float(np.median(null)), 4),
                             null_p90=round(float(np.percentile(null, 90)), 4),
                             perm_p=round(float(pv), 4), n_perm=N_PERM,
                             n_repeat=N_REPEAT))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'rf_kfold.tsv', sep='\t', index=False)
    print('\n' + d[['tissue', 'variant', 'auc', 'auc_sd', 'null_median',
                    'perm_p']].to_string(index=False))


if __name__ == '__main__':
    main()
