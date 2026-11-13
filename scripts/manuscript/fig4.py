#!/usr/bin/env python3
"""Figure 4 - a classifier cannot tell the groups apart."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_5'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import mstyle as M
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from fig_rf_loocv import boot_roc, GRID

S5 = gg2.PLOTS / 'stage_5'
COL = {'lens': M.C_LENS, 'tear': M.C_TEAR}


def roc_panel(ax, tis):
    d = pd.read_csv(S5 / f'rf_kfold_oof_all-features_{tis}.tsv', sep='\t')
    y = (d.truth == 'XFG').astype(int).values
    p = d.prob_XFG.values
    fpr, tpr, _ = roc_curve(y, p)
    kf = pd.read_csv(S5 / 'rf_kfold.tsv', sep='\t')
    auc = float(kf[(kf.tissue == tis) & (kf.variant == 'all-features')].auc.iloc[0])
    lo, hi, clo, chi = boot_roc(y, p)
    ax.fill_between(GRID, lo, hi, color=COL[tis], alpha=.16, lw=0, zorder=1)
    ax.plot([0, 1], [0, 1], ls='--', lw=.7, color=M.GREY, zorder=2)
    ax.plot(fpr, tpr, color=COL[tis], lw=1.3, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.text(.96, .06, f'AUC {auc:.3f}\n95% CI {clo:.2f}–{chi:.2f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7,
            linespacing=1.5, color=M.INK)
    ax.set_title(f'{tis.capitalize()} (n = {len(d)})', fontsize=8, pad=3)
    print(f'  {tis} 5-fold: AUC {auc:.3f} (95% CI {clo:.2f}-{chi:.2f})')


def importance_profile(tis, n_shuffle=40, n_trees=500, seed=0):
    cache = S5 / f'rf_importance_profile_{tis}.tsv'
    if cache.exists():
        return pd.read_csv(cache, sep='\t')
    import de
    from sklearn.ensemble import RandomForestClassifier
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    sub, samples = de.filtered(m, md, tis)
    X = np.log2((sub / sub.sum() * 1e6).T.values + 1)
    y = (md.loc[samples, 'disease'] == 'diseased').astype(int).values

    def forest():
        return RandomForestClassifier(n_estimators=n_trees, random_state=seed,
                                      class_weight='balanced', n_jobs=-1)
    obs = np.sort(forest().fit(X, y).feature_importances_)[::-1]
    rng = np.random.default_rng(seed)
    prof = np.array([np.sort(forest().fit(X, rng.permutation(y)).feature_importances_)[::-1]
                     for _ in range(n_shuffle)])
    d = pd.DataFrame(dict(rank=np.arange(1, len(obs) + 1), observed=obs,
                          lo=np.percentile(prof, 5, axis=0),
                          hi=np.percentile(prof, 95, axis=0),
                          med=np.median(prof, axis=0)))
    d.to_csv(cache, sep='\t', index=False)
    return d


def imp_panel(ax, tis):
    d = importance_profile(tis)
    n = len(d)
    ax.fill_between(d['rank'], d.lo, d.hi, color='#cfcfcf', alpha=.9, lw=0, zorder=1,
                    label='shuffled labels, 5–95%')
    ax.bar(d['rank'], d.observed, width=1.0, color=COL[tis], lw=0, zorder=2,
           label='observed')
    ax.plot(d['rank'], d.hi, color='#4a4a4a', lw=.9, ls='--', zorder=3)
    ax.set_xlim(0, n + 1); ax.set_ylim(0, None)
    ax.set_xlabel('miRNA, ranked by importance')
    ax.set_ylabel('Gini importance')
    ax.set_title(f'{tis.capitalize()} ({n} miRNAs)', fontsize=8, pad=3)
    ax.legend(loc='upper right', handlelength=1.1, handletextpad=.4, borderpad=.2,
              labelspacing=.25)
    print(f'  {tis} importance: {int((d.observed > d.hi).sum())} of {n} above the '
          f'shuffled 95th percentile')


def main():
    M.use()
    fig, axes = plt.subplots(2, 2, figsize=(M.W2, M.W2 * 0.82),
                             gridspec_kw=dict(wspace=.30, hspace=.42, left=.085,
                                              right=.985, top=.945, bottom=.085))
    for ax, t in zip(axes[0], ('lens', 'tear')):
        roc_panel(ax, t)
    for ax, t in zip(axes[1], ('lens', 'tear')):
        imp_panel(ax, t)
    for ax in axes.ravel():
        ax.grid(alpha=.16, lw=.5); ax.set_axisbelow(True)
    # the ROC panels' aspect='equal' shrinks their axes box, so the letters are
    # placed in figure coordinates to line up down the plate.
    fig.canvas.draw()
    for (ax, L), (col, row) in zip(zip(axes.ravel(), 'abcd'),
                                   [(0, 0), (1, 0), (0, 1), (1, 1)]):
        bb = ax.get_position()
        x = min(axes[r][col].get_position().x0 for r in (0, 1)) - .058
        fig.text(x, bb.y1 + .022, L, fontsize=plt.rcParams['font.size'],
                 fontweight='bold', va='top', ha='left', color=M.INK)
    M.save(fig, 'figure4')


if __name__ == '__main__':
    main()
