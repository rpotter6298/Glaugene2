#!/usr/bin/env python3
"""Replication A+ variant - Jones concordance on mean/SD (signal-to-noise) instead of mean; both metrics side by side."""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'CBGM-v22')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = gg2.PLOTS / 'replication'
GEO = gg2.ROOT / 'refs' / 'GSE291021_jones2026_tear_counts.csv.gz'
C_OK, C_TOP = '#3a6ea5', '#c1443f'
MIN_DET = 0.5      # detected in >=50% of samples in both cohorts (Jones' core rule)


def stats_for(x, y):
    return (stats.pearsonr(np.log10(x), np.log10(y)).statistic,
            stats.spearmanr(x, y).statistic)


def main():
    J = pd.read_csv(GEO, index_col=0)
    md, m = gg2.metadata(), gg2.counts(gg2.PRIMARY)
    tear = [c for c in m.columns if md.loc[c, 'tissue_name'] == 'tear']
    jcpm = J / J.sum() * 1e6
    ocpm = m[tear] / m[tear].sum() * 1e6

    d = pd.DataFrame({
        'theirs_mean': jcpm.mean(axis=1), 'theirs_sd': jcpm.std(axis=1),
        'ours_mean': ocpm.mean(axis=1), 'ours_sd': ocpm.std(axis=1)})
    d = d.dropna()
    d['theirs_snr'] = d.theirs_mean / d.theirs_sd
    d['ours_snr'] = d.ours_mean / d.ours_sd
    # mean/SD has a HARD FLOOR for sparsely detected miRNAs: seen in k of n samples,
    # the ratio is fixed by k and n alone, so the SNR panel's bottom-left collapses
    # into detection-frequency stripes, not agreement. Hence the >=MIN_DET-in-BOTH-
    # cohorts filter - the same ">=50% of samples" rule Jones used for their 318 core
    d['det_theirs'] = (J > 0).mean(axis=1)
    d['det_ours'] = (m[tear] > 0).mean(axis=1)
    ok = ((d.theirs_mean > 0) & (d.ours_mean > 0) & (d.theirs_sd > 0) & (d.ours_sd > 0)
          & np.isfinite(d.theirs_snr) & np.isfinite(d.ours_snr))
    b = d[ok]
    core = b[(b.det_theirs >= MIN_DET) & (b.det_ours >= MIN_DET)]
    d.to_csv(OUT / 'tear_jones_meansd.tsv', sep='\t')

    r_m, rho_m = stats_for(b.theirs_mean, b.ours_mean)
    r_s, rho_s = stats_for(core.theirs_snr, core.ours_snr)
    r_mc, rho_mc = stats_for(core.theirs_mean, core.ours_mean)
    # is the scatter actually tighter? residual SD about the diagonal, in dex
    res_m = np.std(np.log10(b.ours_mean) - np.log10(b.theirs_mean))
    res_s = np.std(np.log10(core.ours_snr) - np.log10(core.theirs_snr))
    res_mc = np.std(np.log10(core.ours_mean) - np.log10(core.theirs_mean))
    res_s_all = np.std(np.log10(b.ours_snr) - np.log10(b.theirs_snr))
    print(f'{len(b)} miRNAs non-zero with SD>0 in both cohorts; '
          f'{len(core)} detected in >={MIN_DET:.0%} of samples in both')
    print(f'  all {len(b)}:  mean CPM r={r_m:.3f} rho={rho_m:.3f} scatter {res_m:.2f} dex'
          f'   |  mean/SD scatter {res_s_all:.2f} dex '
          f'(inflated: sparse miRNAs sit on the SNR floor)')
    print(f'  core {len(core)}: mean CPM r={r_mc:.3f} rho={rho_mc:.3f} scatter '
          f'{res_mc:.2f} dex  |  mean/SD r={r_s:.3f} rho={rho_s:.3f} scatter '
          f'{res_s:.2f} dex')

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 7.0))
    # scatter in dex is NOT comparable between the panels (mean/SD compresses a
    # ~7.5 dex axis into ~1.6), so it is also reported as a fraction of the spread
    rng_m = np.log10(b.theirs_mean).max() - np.log10(b.theirs_mean).min()
    rng_s = np.log10(core.theirs_snr).max() - np.log10(core.theirs_snr).min()
    print(f'  relative scatter (scatter / axis range): mean CPM {res_mc / rng_m:.3f}, '
          f'mean/SD {res_s / rng_s:.3f}')
    for ax, (xs, ys, lab, r, rho, res, note) in zip(axes, [
            (b.theirs_mean, b.ours_mean, 'mean CPM', r_m, rho_m, res_m,
             f'all {len(b)} detected in both'),
            (core.theirs_snr, core.ours_snr, 'mean / SD  (signal-to-noise)', r_s, rho_s,
             res_s, f'{len(core)} detected in ≥50% of samples in both')]):
        lo = min(np.log10(xs).min(), np.log10(ys).min()) - .25
        hi = max(np.log10(xs).max(), np.log10(ys).max()) + .25
        ax.plot([lo, hi], [lo, hi], ls='--', lw=1.2, color='#999', zorder=1)
        ax.scatter(np.log10(xs), np.log10(ys), s=13, c=C_OK, alpha=.55,
                   edgecolor='none', zorder=2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel(f'Jones 2026  log10 {lab}', fontsize=10)
        ax.set_ylabel(f'this study  log10 {lab}', fontsize=10)
        ax.set_title(f'{lab}\n{note}', fontsize=12.5, weight='bold', linespacing=1.5)
        rel = res / (rng_m if 'CPM' in lab else rng_s)
        extra = (f'\n\non the 350 core miRNAs:\nr = {r_mc:.3f}  ρ = {rho_mc:.3f}'
                 f'\nscatter {res_mc:.2f} dex' if 'CPM' in lab else '')
        ax.text(.03, .97, f'r = {r:.3f}\nρ = {rho:.3f}\nscatter {res:.2f} dex'
                f'  ({rel:.1%} of range){extra}',
                transform=ax.transAxes, ha='left', va='top', fontsize=9.4,
                linespacing=1.5, bbox=dict(fc='white', ec='#ddd', alpha=.92))
        ax.grid(alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Tear concordance with Jones 2026: abundance vs signal-to-noise',
                 fontsize=16.5, weight='bold', y=.980)
    fig.text(.5, .912, f'{gg2.PRIMARY}  ·  {len(b)} miRNAs detected in both cohorts  ·  '
             f'dividing by the across-sample SD suppresses miRNAs that are abundant '
             f'but unstable  ·  the SNR panel needs the detection filter, see notes',
             ha='center', va='center', fontsize=10.2, color='#555')
    fig.text(.5, .028, 'On the same 350 miRNAs, mean CPM agrees better (r = 0.982 vs '
             '0.760). mean/SD has a smaller absolute scatter only because it compresses '
             'a 7.5-dex axis into 1.6;\nrelative to its own spread it is about three '
             'times looser. The abundance comparison is the one to quote.',
             ha='center', va='center', fontsize=9.6, color='#333', linespacing=1.6)
    fig.tight_layout(rect=[0, .075, 1, .885])
    f = OUT / 'fig_tear_jones_meansd.png'
    fig.savefig(f, dpi=170, facecolor='white')
    print(f'wrote {f}')


if __name__ == '__main__':
    main()
