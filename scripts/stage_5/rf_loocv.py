#!/usr/bin/env python3
"""Stage 5 — random forest XFG vs control, per tissue, leave-one-out CV."""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve

OUT = gg2.PLOTS / 'stage_5'
OUT.mkdir(parents=True, exist_ok=True)
TISSUES = ['lens', 'tear']
N_TREES, N_PERM, NPROC = 500, 200, 22
SEED = 0


def matrix(tissue):
    md = gg2.metadata()
    m = gg2.matrix(gg2.PRIMARY)
    sub, samples = de.filtered(m, md, tissue)
    cpm = sub / sub.sum() * 1e6                 # per-sample: no cross-sample leak
    X = np.log2(cpm.T.values + 1)               # samples x features
    y = (md.loc[samples, 'disease'] == 'diseased').astype(int).values
    return X, y, list(sub.index), samples


def loocv(X, y, seed=SEED):
    """Pooled out-of-fold probabilities from leave-one-out CV."""
    p = np.empty(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        clf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                     class_weight='balanced', n_jobs=1)
        clf.fit(X[tr], y[tr])
        p[i] = clf.predict_proba(X[i:i + 1])[0, 1]
    return p


def main():
    rows, oof_rows, imp_rows = [], [], []
    for tis in TISSUES:
        X, y, feats, samples = matrix(tis)
        print(f'\n=== {tis}: {X.shape[1]} features x {X.shape[0]} samples '
              f'({y.sum()} XFG / {(1 - y).sum()} control) ===')

        p = loocv(X, y)
        auc = roc_auc_score(y, p)
        pred = (p >= .5).astype(int)
        acc = (pred == y).mean()
        sens = pred[y == 1].mean()
        spec = 1 - pred[y == 0].mean()
        print(f'  observed AUC {auc:.3f}   accuracy {acc:.1%}   '
              f'sens {sens:.1%}   spec {spec:.1%}', flush=True)

        print(f'  running {N_PERM} label permutations (full LOOCV each)...', flush=True)
        rng = np.random.default_rng(SEED)
        perms = [rng.permutation(y) for _ in range(N_PERM)]
        null = Parallel(n_jobs=NPROC)(
            delayed(lambda yy, s: roc_auc_score(yy, loocv(X, yy, seed=s)))(yy, 100 + k)
            for k, yy in enumerate(perms))
        null = np.array(null)
        pval = (1 + (null >= auc).sum()) / (1 + len(null))
        print(f'  permutation null: median AUC {np.median(null):.3f}  '
              f'p90 {np.percentile(null, 90):.3f}   -> perm p = {pval:.3f}', flush=True)

        rows.append(dict(tissue=tis, n=X.shape[0], n_features=X.shape[1],
                         n_case=int(y.sum()), n_control=int((1 - y).sum()),
                         auc=round(auc, 4), accuracy=round(acc, 4),
                         sensitivity=round(sens, 4), specificity=round(spec, 4),
                         null_median_auc=round(float(np.median(null)), 4),
                         null_p90_auc=round(float(np.percentile(null, 90)), 4),
                         perm_p=round(float(pval), 4), n_perm=N_PERM))
        for s, yy, pp in zip(samples, y, p):
            oof_rows.append(dict(tissue=tis, sample=s,
                                 truth='XFG' if yy else 'control',
                                 prob_XFG=round(float(pp), 4)))
        np.save(OUT / f'null_auc_{tis}.npy', null)

        clf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED,
                                     class_weight='balanced', n_jobs=NPROC).fit(X, y)
        for f, v in sorted(zip(feats, clf.feature_importances_),
                           key=lambda t: -t[1])[:25]:
            imp_rows.append(dict(tissue=tis, entry=f, importance=round(float(v), 6)))

    pd.DataFrame(rows).to_csv(OUT / 'rf_loocv.tsv', sep='\t', index=False)
    pd.DataFrame(oof_rows).to_csv(OUT / 'rf_loocv_oof.tsv', sep='\t', index=False)
    pd.DataFrame(imp_rows).to_csv(OUT / 'rf_importance.tsv', sep='\t', index=False)
    print(f'\nwrote {OUT}/rf_loocv.tsv, rf_loocv_oof.tsv, rf_importance.tsv')
    print(pd.DataFrame(rows)[['tissue', 'auc', 'accuracy', 'null_median_auc',
                              'perm_p']].to_string(index=False))


if __name__ == '__main__':
    main()
