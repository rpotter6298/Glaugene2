"""Stage 3 — normalisation families: relative (composition) vs absolute (anchored to wet-lab RNA input)."""
import numpy as np, pandas as pd


def sf_median_of_ratios(counts):
    """DESeq2 size factors: median ratio to the geometric-mean reference sample."""
    log = np.log(counts.replace(0, np.nan))
    ref = log.mean(axis=1)
    ok = ref.notna() & np.isfinite(ref)
    sf = np.exp((log[ok.values].sub(ref[ok], axis=0)).median(axis=0))
    return sf / np.exp(np.log(sf).mean())


def sf_tmm(counts, trim_m=0.30, trim_a=0.05):
    """edgeR TMM: trimmed mean of M-values against a reference sample."""
    cpm = counts / counts.sum()
    ref_i = (cpm.sub(cpm.mean(axis=1), axis=0) ** 2).sum().idxmin()
    ref, nref = counts[ref_i].values, counts[ref_i].sum()
    f = {}
    for c in counts.columns:
        obs, nobs = counts[c].values, counts[c].sum()
        keep = (obs > 0) & (ref > 0)
        o, r = obs[keep] / nobs, ref[keep] / nref
        M = np.log2(o / r)
        A = 0.5 * np.log2(o * r)
        w = (nobs - obs[keep]) / (nobs * obs[keep]) + (nref - ref[keep]) / (nref * ref[keep])
        good = np.isfinite(M) & np.isfinite(A) & np.isfinite(w) & (w > 0)
        M, A, w = M[good], A[good], 1 / w[good]
        loM, hiM = np.quantile(M, [trim_m, 1 - trim_m])
        loA, hiA = np.quantile(A, [trim_a, 1 - trim_a])
        sel = (M >= loM) & (M <= hiM) & (A >= loA) & (A <= hiA)
        f[c] = 2 ** (np.sum(w[sel] * M[sel]) / np.sum(w[sel])) if sel.sum() else 1.0
    f = pd.Series(f)
    f = f / np.exp(np.log(f).mean())
    lib = counts.sum() * f
    return lib / np.exp(np.log(lib).mean())


def sf_cpm(counts):
    lib = counts.sum()
    return lib / np.exp(np.log(lib).mean())


def sf_total_reads(counts, total_reads):
    """DEPRECATED: scale by total sequenced reads — tear libraries are 4.9-71.9% tRNA, so the denominator moves with tRNA; superseded by sf_rna_input."""
    tr = total_reads.reindex(counts.columns).astype(float)
    return tr / np.exp(np.log(tr).mean())


def sf_rna_input(counts, total_reads, rna_ng):
    """ABSOLUTE: size factor = reads per ng input RNA (gg2.rna_input())."""
    tr = total_reads.reindex(counts.columns).astype(float)
    ng = rna_ng.reindex(counts.columns).astype(float)
    sf = tr / ng
    return sf / np.exp(np.log(sf).mean())


# CPM is PRIMARY per the 2026 benchmark (NBSR uses the same denominator internally);
# TMM and median-of-ratios are kept for comparability with Genes 2022 and the tear
# miRNome paper; input-RNA asks the different question "did the pool shrink".
FAMILIES = {
    'CPM':              ('relative', sf_cpm),
    'TMM':              ('relative', sf_tmm),
    'median-of-ratios': ('relative', sf_median_of_ratios),
    'input-RNA':        ('absolute', None),                  # handled by caller
}
PRIMARY = 'CPM'


def size_factors(name, counts, total_reads=None, rna_ng=None):
    if name == 'input-RNA':
        return sf_rna_input(counts, total_reads, rna_ng)
    if name == 'total-reads':
        return sf_total_reads(counts, total_reads)
    return FAMILIES[name][1](counts)
