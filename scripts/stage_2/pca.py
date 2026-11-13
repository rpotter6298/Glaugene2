#!/usr/bin/env python3
"""Stage 2 — shared PCA helpers (figures built by pca_tissue_depth.py and pca_disease.py)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

# PCA rather than UMAP: at n=95 with one dominant factor the variance explained is
# interpretable; UMAP's neighbour parameter manufactures apparent clusters.
OUT = gg2.PLOTS / 'stage_2'; OUT.mkdir(parents=True, exist_ok=True)
ARM = gg2.PRIMARY
C_LENS, C_TEAR = '#c98a3f', '#3a6ea5'
C_CTRL, C_DIS = '#5b8c5a', '#c1443f'


def pca(counts, min_frac=0.20, keep=None):
    """log2 CPM -> centred -> SVD. Returns scores, variance ratios, n entries."""
    m = counts
    if keep is None:
        keep = (m >= 10).sum(axis=1) >= min_frac * m.shape[1]
    else:
        keep = m.index.isin(list(keep))
    cpm = m[keep] / m[keep].sum() * 1e6
    X = np.log2(cpm + 1).T
    X = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(X.values, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    return pd.DataFrame(U[:, :4] * S[:4], index=X.index,
                        columns=[f'PC{i+1}' for i in range(4)]), var, int(keep.sum())


def d_between(a, b):
    return abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)



def rarefy(counts, n=None, seed=0):
    """Subsample every sample to the same number of reads."""
    rng = np.random.default_rng(seed)
    n = n or int(counts.sum().min())
    return pd.DataFrame(
        {c: rng.multinomial(n, counts[c].values / counts[c].sum()) for c in counts.columns},
        index=counts.index), n


def align_signs(P, ref):
    """Flip each component to point the same way as `ref`; SVD sign is arbitrary."""
    P = P.loc[ref.index].copy()
    for c in P.columns:
        if np.corrcoef(P[c], ref[c])[0, 1] < 0:
            P[c] = -P[c]
    return P
