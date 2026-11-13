#!/usr/bin/env python3
"""Supplementary Table S4 - each published criterion applied to this cohort, with provenance.

Output -> manuscript/final_pass/table_S4_published_criteria.tsv
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
os.environ.setdefault('GG2_ARM', 'BEA-dedup')
import gg2
import pandas as pd

OUT = gg2.ROOT / 'manuscript' / 'final_pass' / 'table_S4_published_criteria.tsv'

# study, design, criterion as printed, reported, test
CRIT = [
    ('Tomczyk-Socha 2022', 'lens capsule, 5 v 5', 'P < 0.1*', '6',
     lambda s: s.ql_p < .10),
    ('Rao 2020', 'peripheral blood, 3 v 3 (XFG against XFS with ocular hypertension)',
     'P < 0.05*', '12', lambda s: s.ql_p < .05),
    ('Kosior-Jarecka 2021', 'aqueous humour, 42 v 36, 22-miRNA panel',
     'unadjusted P < 0.05', '0', lambda s: s.ql_p < .05),
    ('Drewry 2018', 'aqueous humour, 12 POAG, 12 XFG v 11',
     '|log2FC| ≥ 0.6, unadjusted P ≤ 0.05, and detected in ≥ 5 samples',
     '5 (XFG)',
     lambda s: (s.ql_p <= .05) & (s.ql_log2FC.abs() >= .6) & (s.n_det >= 5)),
    ('Cho 2022', 'aqueous humour, 11 XFG, 8 NTG v 7',
     'fold change > 2 and unadjusted P < 0.05', '12 (XFG)',
     lambda s: (s.ql_p < .05) & (s.ql_log2FC.abs() > 1.)),
    # the fold-change half of Czop's criterion is non-binding; see fig_published_criteria
    ('Czop 2023', 'aqueous humour, 9 v 9', 'unadjusted P < 0.01', '20',
     lambda s: s.ql_p < .01),
]

# row of refs/glaucoma_mirna_studies.tsv each study corresponds to
REF_KEY = {'Tomczyk-Socha 2022': 'Tomczyk-Socha et al.', 'Rao 2020': 'Rao et al.',
           'Kosior-Jarecka 2021': 'Kosior-Jarecka et al.', 'Drewry 2018': 'Drewry et al.',
           'Cho 2022': 'Cho et al.', 'Czop 2023': 'Czop et al.'}

# written out from the refs table's shorthand; the provenance column says where from
CORRECTION = {
    'Tomczyk-Socha 2022': 'Not stated anywhere in the paper',
    'Rao 2020': 'Not stated',
    'Kosior-Jarecka 2021': 'None on the disease comparison; Benjamini-Hochberg applied '
                           'only to the enrichment analysis',
    'Drewry 2018': 'None on the disease comparison; Benjamini-Hochberg and a bias '
                   'correction applied only to the pathway analysis',
    'Cho 2022': 'None on the disease comparison; false discovery rate applied only to '
                'pathway enrichment',
    'Czop 2023': 'None on the disease comparison; Benjamini-Hochberg applied only to '
                 'the enrichment analysis',
}


def main():
    md, mx = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    de = pd.read_csv(gg2.PLOTS / 'stage_3' / 'de_edger.tsv', sep='\t')
    de = de[de['norm'] == 'CPM']

    sub = {}
    for t in ('lens', 'tear'):
        cols = [c for c in mx.columns if md.loc[c, 'tissue_name'] == t]
        ndet = (mx[cols] > 0).sum(axis=1)
        x = de[de.tissue == t].copy()
        x['n_det'] = x.entry.map(ndet).fillna(0)
        sub[t] = x

    refs = pd.read_csv(gg2.ROOT / 'refs' / 'glaucoma_mirna_studies.tsv', sep='\t')
    refs = refs.set_index('study')

    rows = []
    for name, design, crit, reported, f in CRIT:
        r = refs.loc[REF_KEY[name]]
        rows.append(dict(
            study=name, design=design, criterion=crit, reported=reported,
            lens=int(f(sub['lens']).sum()), tear=int(f(sub['tear']).sum()),
            correction_in_that_study=CORRECTION[name],
            criterion_verified_from=f'{r.verified.split(",")[0].split(";")[0].strip()}, '
                                    f'doi {r.doi}'))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, sep='\t', index=False)
    print(f'wrote {OUT}  ({len(d)} rows)')
    print(d[['study', 'criterion', 'reported', 'lens', 'tear']].to_string(index=False))


if __name__ == '__main__':
    main()
