#!/usr/bin/env python3
"""Stage 6 — miRNA-level functional enrichment (Godard & van Eyll 2015, Strategy 3)."""
import os
import re
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-dedup')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import de
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = gg2.PLOTS / 'stage_6'
OUT.mkdir(parents=True, exist_ok=True)
REF = gg2.ROOT / 'refs' / 'mieaa'
COLLECTIONS = {'GO:BP': 'miRPathDB_GO_Biological_process_mature.gmt',
               'KEGG': 'miRPathDB_KEGG_mature.gmt',
               'Reactome': 'miRPathDB_Reactome_mature.gmt',
               'WikiPathways': 'miRPathDB_WikiPathways_mature.gmt'}
FIG_COLLECTION = os.environ.get('GG2_MIRCOLL', 'GO:BP')
# GG2_COLLAPSE=seed: one entry per seed family (mature positions 2-8). miRPathDB
# annotates every family member identically, so a family of five casts five votes
# for the same biology; a signal that survives collapsing is not a family-size artefact.
COLLAPSE = os.environ.get('GG2_COLLAPSE', 'none')
SET_MIN, SET_MAX = 5, 200      # miRNAs per set, after restriction to expressed
N_PERM, N_RANDOM, SEED = 2000, 1000, 0
TOP_NS = (25, 50)
DEPTH_BINS = 5
SKIP_ORA = os.environ.get('GG2_SKIP_ORA') == '1'
Q_CUT = 0.05
PSEUDO = 1.0
# GSEA weight exponent: 1 = classic weighted KS; 0 = unweighted (rank only). The tear
# tail is far longer (log2fc to -11 vs +4.9), so p=1 weights it more; p=0 checks a
# one-sided result is not just the longer tail.
GSEA_P = float(os.environ.get('GG2_GSEA_P', '1'))
SUFFIX_P = '' if GSEA_P == 1 else f'_p{GSEA_P:g}'
SUFFIX = ('_seed' if COLLAPSE == 'seed' else '') + SUFFIX_P
C_LENS, C_TEAR = '#3a6ea5', '#c1443f'
EYE = re.compile(r'lens|eye|cornea|visual|retina|photorecep|lacrimal', re.I)


def bh(p):
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def load_gmt(path):
    sets = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            sets[p[0]] = set(p[2:]) - {''}
    return sets


# Defaults to the 42 patients who gave BOTH tissues; with 42 pairs available there is
# no need to assume the 10 single-tissue patients match. GG2_PAIRED=0 to include all.
PAIRED = os.environ.get('GG2_PAIRED', '1') != '0'


def paired_samples(md, cols):
    """Columns belonging to patients contributing both lens and tear."""
    sub = md.loc[[c for c in cols if md.loc[c, 'tissue_name'] in ('lens', 'tear')]]
    n = sub.groupby('patient').tissue_name.nunique()
    both = set(n.index[n == 2])
    return [c for c in sub.index if sub.loc[c, 'patient'] in both]


def profile():
    """Mean CPM per tissue for every miRNA expressed (Stage 1 filter) in either."""
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    if PAIRED:
        m = m[paired_samples(md, m.columns)]
    keep = set()
    for tis in ['lens', 'tear']:
        sub, _ = de.filtered(m, md, tis); keep |= set(sub.index)
    cpm = m / m.sum() * 1e6
    out = {}
    for tis in ['lens', 'tear']:
        s = [c for c in m.columns if md.loc[c, 'tissue_name'] == tis]
        out[tis] = cpm.loc[sorted(keep), s].mean(axis=1)
    d = pd.DataFrame(out)
    if COLLAPSE == 'seed':
        from tissue_target_enrichment import seeds
        sd = seeds()
        d['seed'] = [sd.get(x, x) for x in d.index]
        # family named after its most abundant member (either tissue)
        rep = d.assign(tot=d.lens + d.tear).sort_values('tot', ascending=False)
        rep = rep[~rep.seed.duplicated()]
        name = dict(zip(rep.seed, rep.index))
        agg = d.groupby('seed')[['lens', 'tear']].sum()
        agg['members'] = d.groupby('seed').apply(
            lambda g: ','.join(x.replace('hsa-', '') for x in g.index))
        agg.index = [name[sv] for sv in agg.index]
        SEED_MAP.update({mi: name[sv] for mi, sv in zip(d.index, d.seed)})
        d = agg
    d['log2fc'] = np.log2((d.lens + PSEUDO) / (d.tear + PSEUDO))
    return d.sort_values('log2fc', ascending=False)


SEED_MAP = {}   # miRNA -> family representative, filled by profile() when collapsing


def collapse_sets(sets):
    """Collapse GMT sets onto expressed seed families; for GSEA, not a genome-wide ORA denominator."""
    if COLLAPSE != 'seed':
        return sets
    return {k: {SEED_MAP[x] for x in v if x in SEED_MAP} for k, v in sets.items()}


GLOBAL_SEED_MAP = {}


def global_seed_map():
    """miRNA -> family representative, over every miRNA miRBase gives a seed for."""
    if GLOBAL_SEED_MAP:
        return GLOBAL_SEED_MAP
    from tissue_target_enrichment import seeds
    sd = seeds()
    by_seed = {}
    for mi, sv in sd.items():
        by_seed.setdefault(sv, []).append(mi)
    named = {sd.get(mi, mi): rep for mi, rep in SEED_MAP.items()}
    for sv, members in by_seed.items():
        rep = named.get(sv) or sorted(members)[0]
        for mi in members:
            GLOBAL_SEED_MAP[mi] = rep
    return GLOBAL_SEED_MAP


def collapse_sets_global(sets):
    """Collapse GMT sets onto ALL seed families, expressed here or not."""
    if COLLAPSE != 'seed':
        return sets
    g = global_seed_map()
    return {k: {g.get(x, x) for x in v} for k, v in sets.items()}


# GSEA
def es_all(r, S):
    """Enrichment score for every set at once. r: ranking metric in rank order
    (n,), S: (sets x n) 0/1 membership. Weighted KS, p=GSEA_P."""
    w = np.abs(r) ** GSEA_P
    hit = S * w
    hit = np.cumsum(hit, axis=1) / hit.sum(axis=1, keepdims=True)
    nh = S.sum(axis=1, keepdims=True)
    miss = np.cumsum(1 - S, axis=1) / (S.shape[1] - nh)
    dev = hit - miss
    i = np.argmax(np.abs(dev), axis=1)
    return dev[np.arange(len(dev)), i]


def gsea(prof, sets, label):
    names = list(prof.index); n = len(names)
    r = prof.log2fc.values
    keep = {k: [names.index(x) for x in v if x in set(names)] for k, v in sets.items()}
    keep = {k: v for k, v in keep.items() if SET_MIN <= len(v) <= SET_MAX}
    terms = list(keep)
    S = np.zeros((len(terms), n))
    for i, t in enumerate(terms):
        S[i, keep[t]] = 1
    obs = es_all(r, S)
    rng = np.random.default_rng(SEED)
    null = np.empty((N_PERM, len(terms)))
    for k in range(N_PERM):
        perm = rng.permutation(n)
        null[k] = es_all(r, S[:, perm])     # shuffle membership over the ranking
    # NES: divide by mean |null ES| of the same sign, per set
    pos = np.where(null > 0, null, np.nan); neg = np.where(null < 0, null, np.nan)
    mpos = np.nanmean(pos, axis=0); mneg = np.nanmean(neg, axis=0)
    nes = np.where(obs >= 0, obs / mpos, -obs / mneg)
    nes_null = np.where(null >= 0, null / mpos, -null / mneg)
    p_emp = np.where(obs >= 0, (null >= obs).mean(axis=0), (null <= obs).mean(axis=0))
    p_emp = (p_emp * N_PERM + 1) / (N_PERM + 1)
    # GSEA-style FDR from the pooled null NES
    q_gsea = np.empty(len(terms))
    nn = nes_null.ravel(); nn = nn[np.isfinite(nn)]
    for i, v in enumerate(nes):
        if v >= 0:
            a = (nn >= v).sum() / max((nn >= 0).sum(), 1)
            b = (nes >= v).sum() / max((nes >= 0).sum(), 1)
        else:
            a = (nn <= v).sum() / max((nn < 0).sum(), 1)
            b = (nes <= v).sum() / max((nes < 0).sum(), 1)
        q_gsea[i] = min(a / b, 1.0) if b > 0 else 1.0
    d = pd.DataFrame(dict(collection=label, term=terms,
                          n_mirnas=[len(keep[t]) for t in terms],
                          es=obs, nes=nes, p_perm=p_emp, q_gsea=q_gsea,
                          q_bh=bh(p_emp)))
    d['leaning'] = np.where(d.nes > 0, 'lens', 'tear')
    d['eye_term'] = d.term.str.contains(EYE)
    d['leading_edge'] = [', '.join(x.replace('hsa-', '') for x in
                          leading_edge(r, names, keep[t], obs[i]))
                         for i, t in enumerate(terms)]
    return d.sort_values('p_perm')


def leading_edge(r, names, members, es):
    m = np.zeros(len(r)); m[members] = 1
    w = np.abs(r) ** GSEA_P; hit = np.cumsum(m * w) / (m * w).sum()
    miss = np.cumsum(1 - m) / (len(r) - m.sum()); dev = hit - miss
    i = int(np.argmax(np.abs(dev)))
    idx = [j for j in members if (j <= i if es >= 0 else j >= i)]
    return [names[j] for j in sorted(idx, key=lambda j: -abs(r[j]))][:8]


# ORA
def ora(prof, sets, label):
    rows, cal = [], []
    for tis in ['lens', 'tear']:
        ranked = prof[tis].sort_values(ascending=False)
        annotated = set().union(*sets.values())
        universe = [x for x in ranked.index if x in annotated]
        U = set(universe); N = len(U)
        usable = {k: v & U for k, v in sets.items()}
        usable = {k: v for k, v in usable.items() if SET_MIN <= len(v) <= SET_MAX}
        rng = np.random.default_rng(SEED)
        # Abundant miRNAs are better annotated (Spearman log10 CPM vs term count
        # +0.21 lens, +0.34 tear), so match the null on annotation depth: DEPTH_BINS
        # quantile bins of term count, drawn with the observed set's bin composition.
        depth = pd.Series({x: sum(x in v for v in usable.values()) for x in universe})
        dbin = pd.Series(pd.qcut(depth.rank(method='first'), DEPTH_BINS, labels=False),
                         index=depth.index)
        members = {b: list(dbin.index[dbin == b]) for b in range(DEPTH_BINS)}

        def matched_draw(ref):
            out = []
            for b, k in pd.Series([dbin[x] for x in ref]).value_counts().items():
                out += list(rng.choice(members[b], int(k), replace=False))
            return set(out)

        for n_top in TOP_NS:
            top = set(universe[:n_top])

            def counts(mset):
                res = []
                for k, v in usable.items():
                    x = len(mset & v)
                    if x:
                        res.append((k, x, len(v), hypergeom.sf(x - 1, N, len(v), len(mset))))
                d = pd.DataFrame(res, columns=['term', 'k', 'K', 'p'])
                d['q'] = bh(d.p.values) if len(d) else []
                return d

            d = counts(top)
            d.insert(0, 'collection', label); d.insert(1, 'tissue', tis)
            d.insert(2, 'n_top', n_top); d['n_universe'] = N
            rows.append(d.sort_values('p'))
            obs = int((d.q < Q_CUT).sum())
            null = np.array([int((counts(matched_draw(top)).q < Q_CUT).sum())
                             for _ in range(N_RANDOM)])
            pv = (int((null >= obs).sum()) + 1) / (N_RANDOM + 1)
            cal.append(dict(collection=label, tissue=tis, n_top=n_top, n_universe=N,
                            observed=obs, null_median=float(np.median(null)),
                            null_p90=float(np.percentile(null, 90)),
                            null_max=int(null.max()), perm_p=round(pv, 4),
                            n_random=N_RANDOM))
            print(f'    ORA {tis} top{n_top}: {obs} sets q<{Q_CUT}; random sets median '
                  f'{np.median(null):.0f}, p90 {np.percentile(null, 90):.0f}  '
                  f'-> p = {pv:.3f}', flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cal)


def figure(g, prof):
    g = g[g.collection == FIG_COLLECTION]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    for ax, side, col in ((axes[0], 'lens', C_LENS), (axes[1], 'tear', C_TEAR)):
        side_all = g[g.leaning == side]
        s = side_all[side_all.q_gsea < Q_CUT].nsmallest(14, 'p_perm')
        s = s.reindex(s.nes.abs().sort_values().index)   # bars ordered by effect size
        passed = len(s) > 0
        if not passed:
            # Nothing passes: draw the best few in grey so the panel shows how far short they fall
            s = side_all.nsmallest(6, 'p_perm')
            s = s.reindex(s.nes.abs().sort_values().index)
        else:
            pass
        bar_col = col if passed else '#c8c8c8'
        ax.barh(np.arange(len(s)), s.nes.abs(), color=bar_col, height=.72)
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels([(t[:44] + '…' if len(t) > 45 else t).replace('_', ' ')
                            for t in s.term], fontsize=8,
                           color='#222' if passed else '#888')
        for i, (_, r) in enumerate(s.iterrows()):
            ax.text(abs(r.nes) + .03, i, f'q={r.q_gsea:.2f}  n={r.n_mirnas}',
                    va='center', fontsize=7.6, color='#555' if passed else '#999')
        ax.set_xlim(0, max(s.nes.abs().max() * 1.35, 1.0))
        ax.set_ylim(-.7, 13.3)
        if not passed:
            ax.text(.5, .72, f'nothing at FDR<{Q_CUT}\n(best {len(s)} shown)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10.5,
                    color='#888')
        n_sig = int(((g.leaning == side) & (g.q_gsea < Q_CUT)).sum())
        ax.set_title(f'{side}-leaning\n{n_sig} sets at FDR<{Q_CUT}', fontsize=12,
                     weight='bold', loc='left', linespacing=1.5)
        ax.set_xlabel('|NES|', fontsize=9.6)
        ax.grid(axis='x', alpha=.22); ax.set_axisbelow(True)
    fig.suptitle('miRNA-level GSEA, lens vs tear', fontsize=16.5, weight='bold', y=.980)
    fig.text(.5, .922, f'{gg2.PRIMARY}  ·  {FIG_COLLECTION} (miRPathDB, miRNA sets)  ·  '
             f'{len(prof)} expressed ' + ('seed families' if COLLAPSE == 'seed' else 'miRNAs') + ' ranked by log2(lens/tear CPM)  ·  '
             f'{N_PERM:,} rank permutations', ha='center', va='center',
             fontsize=10.2, color='#555')
    fig.tight_layout(rect=[0, .012, 1, .89])
    fig.savefig(OUT / f'fig_mirna_gsea{SUFFIX}.png', dpi=170, facecolor='white')


def main():
    prof = profile()
    print(f'{len(prof)} expressed ' + ('seed families' if COLLAPSE == 'seed' else 'miRNAs')
          + '; top of ranking: '
          + ', '.join(f'{x.replace("hsa-", "")} ({v:+.1f})'
                      for x, v in prof.log2fc.head(4).items())
          + ' … bottom: '
          + ', '.join(f'{x.replace("hsa-", "")} ({v:+.1f})'
                      for x, v in prof.log2fc.tail(3).items()), flush=True)
    prof.to_csv(OUT / f'mirna_profile{SUFFIX}.tsv', sep='\t')
    G, O, Cal = [], [], []
    for label, fn in COLLECTIONS.items():
        sets = collapse_sets(load_gmt(REF / fn))
        print(f'\n=== {label}: {len(sets):,} sets ===', flush=True)
        g = gsea(prof, sets, label)
        for side in ['lens', 'tear']:
            k = int(((g.leaning == side) & (g.q_gsea < Q_CUT)).sum())
            print(f'  GSEA {side}-leaning: {k} sets FDR<{Q_CUT} '
                  f'({int(((g.leaning == side) & (g.q_bh < Q_CUT)).sum())} by BH)', flush=True)
        eye = g[g.eye_term].nsmallest(3, 'p_perm')
        if len(eye):
            print('  eye terms: ' + '; '.join(f'{r.term[:40]} NES {r.nes:+.2f} q {r.q_gsea:.2f}'
                                              for _, r in eye.iterrows()), flush=True)
        G.append(g)
        if not SKIP_ORA:
            o, c = ora(prof, sets, label); O.append(o); Cal.append(c)
    pd.concat(G).to_csv(OUT / f'mirna_gsea{SUFFIX}.tsv', sep='\t', index=False)
    figure(pd.concat(G), prof)
    if not SKIP_ORA:
        pd.concat(O).to_csv(OUT / f'mirna_ora{SUFFIX}.tsv', sep='\t', index=False)
        cal = pd.concat(Cal)
        cal.to_csv(OUT / f'mirna_ora_calibration{SUFFIX}.tsv', sep='\t', index=False)
        print('\n' + cal.to_string(index=False))
    print(f'\nwrote {OUT}/mirna_gsea{SUFFIX}.tsv, mirna_ora{SUFFIX}.tsv, '
          f'mirna_ora_calibration{SUFFIX}.tsv, fig_mirna_gsea{SUFFIX}.png')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--replot':
        # Rebuild the figure from the saved tables; no permutations rerun.
        figure(pd.read_csv(OUT / f'mirna_gsea{SUFFIX}.tsv', sep='\t'),
               pd.read_csv(OUT / f'mirna_profile{SUFFIX}.tsv', sep='\t', index_col=0))
        print(f'replotted fig_mirna_gsea{SUFFIX}.png')
    else:
        main()
