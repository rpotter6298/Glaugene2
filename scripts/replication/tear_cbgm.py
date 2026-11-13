#!/usr/bin/env python3
"""Replication A — the 2026 tear-fluid miRNome paper (CBGM-like arm)."""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'CBGM-v22')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import pandas as pd

ARM = gg2.PRIMARY
OUT = gg2.PLOTS / 'replication'
OUT.mkdir(parents=True, exist_ok=True)

THEIR_TOP5 = ['hsa-let-7a-5p', 'hsa-let-7f-5p', 'hsa-let-7b-5p',
              'hsa-let-7g-5p', 'hsa-let-7c-5p']
THEIR = {'detected': 1110, 'constitutive': 318, 'universal': 246}
THEIR_FAMS = {'let-7': 13, 'miR-30': 10, 'miR-8/200': 10, 'miR-10': 8, 'miR-17': 8}
# seed reference members; miR-8/200 spans two seeds in human (200b/c/429 vs 200a/141)
FAM_REF = {'let-7': ['hsa-let-7a-5p'], 'miR-30': ['hsa-miR-30a-5p'],
           'miR-8/200': ['hsa-miR-200b-3p', 'hsa-miR-200a-3p'],
           'miR-10': ['hsa-miR-10a-5p'], 'miR-17': ['hsa-miR-17-5p']}


def seeds():
    """mature name -> seed (positions 2-8)."""
    fa = gg2.ROOT / 'refs' / 'mature_current.fa'
    out, name = {}, None
    for line in fa.read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]
        elif name:
            s = line.strip().replace('U', 'T')
            if len(s) >= 8:
                out[name] = s[1:8]
            name = None
    return out


def main():
    md = gg2.metadata()
    tear = [c for c in gg2.counts(ARM).columns if md.loc[c, 'tissue_name'] == 'tear']
    ctrl = [c for c in tear if md.loc[c, 'disease'] == 'control']
    SD = seeds()
    rows, abund = [], []

    for pol, m in [('as-published', gg2.counts(ARM)),
                   ('artifact-filtered', gg2.matrix(ARM))]:
        # labels derive from n, not hardcoded: the tear set became 51 when 40A was
        # excluded and a fixed 'all-52' then mislabels every row (noted 2026-10-13,
        # fixed 2026-10-19).
        for cname, cols in [(f'all-{len(tear)}', tear), (f'controls-{len(ctrl)}', ctrl)]:
            t = m[cols]
            n = len(cols)
            det = (t > 0).sum(axis=1)
            present = det[det > 0].index
            core = (det >= n / 2) & (t.mean(axis=1) > 5)     # BOTH, as they did
            got = {'detected': int((det > 0).sum()),
                   'constitutive': int(core.sum()),
                   'universal': int((det[core] == n).sum())}
            for k, v in got.items():
                rows.append(dict(policy=pol, cohort=cname, target=k, theirs=THEIR[k],
                                 ours=v, ratio=round(v / THEIR[k], 2)))

            frac = (t / t.sum()).mean(axis=1).sort_values(ascending=False)
            order = list(frac.index)
            for i, e in enumerate(THEIR_TOP5, 1):
                rank = order.index(e) + 1 if e in order else None
                rows.append(dict(policy=pol, cohort=cname,
                                 target=f'rank of {e.replace("hsa-", "")}',
                                 theirs=i, ours=rank,
                                 ratio=None if rank is None else round(rank / i, 2)))
            if pol == 'as-published':
                for i, (e, v) in enumerate(frac.head(15).items(), 1):
                    abund.append(dict(cohort=cname, rank=i, entry=e,
                                      mean_fraction=round(float(v), 5),
                                      in_their_top5=e in THEIR_TOP5))

            for fam, refs in FAM_REF.items():
                want = {SD[r] for r in refs if r in SD}
                k = sum(1 for e in present if SD.get(e) in want) if want else None
                rows.append(dict(policy=pol, cohort=cname,
                                 target=f'[BLOCKED] family {fam}',
                                 theirs=THEIR_FAMS[fam], ours=k,
                                 ratio=None if k is None else round(k / THEIR_FAMS[fam], 2)))

    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'tear_cbgm.tsv', sep='\t', index=False)
    pd.DataFrame(abund).to_csv(OUT / 'tear_cbgm_abundance.tsv', sep='\t', index=False)

    for pol in ['as-published', 'artifact-filtered']:
        for cname in [f'all-{len(tear)}', f'controls-{len(ctrl)}']:
            s = d[(d.policy == pol) & (d.cohort == cname)]
            print(f'\n=== {pol} · {cname} ===')
            print(f'{"target":26s}{"theirs":>8}{"ours":>8}{"ratio":>8}')
            for _, r in s.iterrows():
                o = '—' if pd.isna(r.ours) else int(r.ours)
                rt = '' if pd.isna(r.ratio) else f'{r.ratio:.2f}'
                print(f'{r.target:26s}{int(r.theirs):>8}{o:>8}{rt:>8}')
    print(f'\nwrote {OUT}/tear_cbgm.tsv and tear_cbgm_abundance.tsv')


if __name__ == '__main__':
    main()
