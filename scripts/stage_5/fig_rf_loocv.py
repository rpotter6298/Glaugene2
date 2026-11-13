#!/usr/bin/env python3
"""Stage 5 — the random-forest ROC figure."""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-raw')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

N_BOOT, BOOT_SEED = 2000, 0
GRID = np.linspace(0, 1, 101)


def boot_roc(y, prob, n_boot=N_BOOT, seed=BOOT_SEED):
    """Stratified patient bootstrap: 95% TPR band on a common FPR grid and 95% AUC CI."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    tprs, aucs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        yy, pp = y[idx], prob[idx]
        if yy.min() == yy.max():
            continue
        f, t, _ = roc_curve(yy, pp)
        tprs.append(np.interp(GRID, f, t))
        aucs.append(roc_auc_score(yy, pp))
    tprs, aucs = np.array(tprs), np.array(aucs)
    return (np.percentile(tprs, 2.5, axis=0), np.percentile(tprs, 97.5, axis=0),
            np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))

OUT = gg2.PLOTS / 'stage_5'
TISSUES = ['lens', 'tear']
C_ROC, C_NULL = '#3a6ea5', '#b9c4cc'


def main():
    res = pd.read_csv(OUT / 'rf_loocv.tsv', sep='\t').set_index('tissue')
    oof = pd.read_csv(OUT / 'rf_loocv_oof.tsv', sep='\t')
    kf = pd.read_csv(OUT / 'rf_kfold.tsv', sep='\t')
    kf = kf[kf.variant == 'all-features'].set_index('tissue')

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.6))
    for i, tis in enumerate(TISSUES):
        r = res.loc[tis]
        panels = [
            ('leave-one-out', oof[oof.tissue == tis].prob_XFG.values,
             (oof[oof.tissue == tis].truth == 'XFG').astype(int).values,
             r.auc, r.accuracy, r.null_median_auc, r.null_p90_auc, r.perm_p,
             int(r.n_perm), None),
        ]
        k = pd.read_csv(OUT / f'rf_kfold_oof_all-features_{tis}.tsv', sep='\t')
        kr = kf.loc[tis]
        panels.append(('repeated 5-fold', k.prob_XFG.values,
                       (k.truth == 'XFG').astype(int).values,
                       kr.auc, None, kr.null_median, kr.null_p90, kr.perm_p,
                       int(kr.n_perm), int(kr.n_repeat)))

        for j, (label, prob, y, auc, acc, nmed, np90, pv, nperm, nrep) in enumerate(panels):
            ax = axes[i, j]
            fpr, tpr, _ = roc_curve(y, prob)
            lo_t, hi_t, lo_a, hi_a = boot_roc(y, prob)
            ax.plot([0, 1], [0, 1], ls='--', lw=1.1, color='#999')
            ax.fill_between(GRID, lo_t, hi_t, color=C_ROC, alpha=.16, lw=0, zorder=1)
            ax.plot(fpr, tpr, lw=2.4, color=C_ROC, zorder=2)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            scheme = 'LOOCV' if j == 0 else f'{nrep}\u00d7 5-fold CV'
            ax.set_title(f'{tis.capitalize()}  |  {scheme}', fontsize=12.5,
                         weight='bold')
            ax.add_patch(plt.Rectangle((.40, .035), .585, .165, transform=ax.transAxes,
                                       fc='white', ec='#ccc', alpha=.92, zorder=2.5))
            ax.text(.96, .115, f'AUC {auc:.3f}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=12, weight='bold',
                    color=C_ROC, zorder=3)
            ax.text(.96, .05, f'95% CI {lo_a:.2f}\u2013{hi_a:.2f}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                    color='#555', zorder=3)
            print(f'    {tis} {scheme}: AUC {auc:.3f} '
                  f'(95% CI {lo_a:.3f}-{hi_a:.3f})', flush=True)
            if i == 1:
                ax.set_xlabel('false positive rate', fontsize=9.6)
            if j == 0:
                ax.set_ylabel('true positive rate', fontsize=9.6)
            ax.grid(alpha=.2); ax.set_axisbelow(True)

    ns = '  ·  '.join(f'{t} n={int(res.loc[t].n)} '
                      f'({int(res.loc[t].n_case)} XFG / {int(res.loc[t].n_control)} '
                      f'control)' for t in TISSUES)
    fig.suptitle('Random forest, XFG vs control', fontsize=16.5, weight='bold', y=.978)
    fig.text(.5, .942, f'{gg2.PRIMARY}  ·  all retained miRNAs, no feature selection  '
             f'·  pooled out-of-fold predictions', ha='center',
             va='center', fontsize=10.5, color='#555')
    fig.text(.5, .918, ns + '  ·  band and 95% CI: 2,000 stratified bootstraps '
             'over patients', ha='center', va='center', fontsize=9.4, color='#777')
    fig.tight_layout(rect=[0, .012, 1, .905])
    fig.savefig(OUT / 'fig_rf_loocv.png', dpi=160, facecolor='white')
    print(f'wrote {OUT}/fig_rf_loocv.png')


if __name__ == '__main__':
    main()
