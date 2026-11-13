#!/usr/bin/env python3
"""Supplementary Table S6 - every miRBase entry excluded, and why.

Four headings: absent (18), inflated (9), annotation (1), unmeasurable (8), plus the six
positive controls and miR-4485-5p, which are retained and flagged. Cross-arm evidence is
given as pooled CPM per pipeline rather than a single fold change; a blank means that
pipeline does not carry the entry at all.

Output -> manuscript/final_pass/table_S6_excluded_entries.tsv
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
os.environ.setdefault('GG2_ARM', 'BEA-dedup')
import gg2
import numpy as np
import pandas as pd

OUT = gg2.ROOT / 'manuscript' / 'final_pass' / 'table_S6_excluded_entries.tsv'
COMPARATORS = ['nfcore', 'CBGM-v22', 'PEXlens-v22']

REASON = {
    'absent': 'mature sequence absent from the reads',
    'inflated': 'counts inflated relative to every other pipeline',
    'annotation': 'sequence present, but almost only inside longer transcripts',
    'unmeasurable': 'reported zero in every sample; this pipeline cannot measure it',
    'control': 'positive control, retained in the analysis',
    'unresolved': 'too short (16 nt) for the read-level test; retained with a flag',
}


def cpm_by_entry(arm):
    """Pooled CPM per entry for one arm: total counts over total reads, x 1e6.
    """
    m = gg2.counts(arm, drop_samples=False)
    return m.sum(axis=1) / m.values.sum() * 1e6


def main():
    arms = [gg2.PRIMARY, 'BEA-raw'] + COMPARATORS
    cpm = pd.DataFrame({a: cpm_by_entry(a) for a in dict.fromkeys(arms)})
    comp = cpm[COMPARATORS]
    # miRBase v23 renamed some entries; map so the arms are comparable by name
    nm = gg2.name_map_v22_v23()
    for a in COMPARATORS:
        for v22, hit in nm.items():
            v23 = hit[0] if isinstance(hit, (list, tuple)) else hit
            if v23 and v23 in comp.index and v22 not in comp.index:
                comp.loc[v22, a] = comp.loc[v23, a]

    verified = pd.read_csv(gg2.SHARED / 'stage_1' / 'suspects_verified.tsv',
                           sep='\t').set_index('entry')

    groups = [('absent', gg2.ARTIFACTS_ABSENT),
              ('inflated', gg2.ARTIFACTS_INFLATED),
              ('annotation', gg2.ARTIFACTS_ANNOTATION),
              ('unmeasurable', list(gg2.BLIND_SPOTS[gg2.PRIMARY])),
              ('unresolved', list(gg2.UNRESOLVED)),
              ('control', list(verified.index[verified.kind == 'control']))]

    rows = []
    for kind, entries in groups:
        for e in entries:
            v = verified.loc[e] if e in verified.index else None
            cp = {f'cpm_{a}': (round(float(cpm.loc[e, a]), 3)
                               if e in cpm.index and not pd.isna(cpm.loc[e, a])
                               else None)
                  for a in cpm.columns}
            rows.append(dict(
                entry=e,
                category=kind,
                reason=REASON[kind],
                nt=int(v.nt) if v is not None and not pd.isna(v.nt) else None,
                reads_assigned=int(v.assigned) if v is not None else None,
                exact_occurrences=int(v.exact) if v is not None else None,
                observed_over_assigned=round(float(v.ratio), 4) if v is not None else None,
                **cp,
                excluded=kind not in ('control', 'unresolved')))
    d = pd.DataFrame(rows)
    d.to_csv(OUT, sep='\t', index=False)

    n_ex = int(d.excluded.sum())
    print(f'wrote {OUT}  ({len(d)} rows, {n_ex} excluded)')
    print(d.groupby('category').size().to_string())
    inf = d[d.category == 'inflated']
    print('\n  inflated entries, pooled CPM by pipeline:')
    print(inf.set_index('entry')[[c for c in d.columns
                                  if c.startswith('cpm_')]].to_string())
    print(f'  excluded total {n_ex}: '
          f'{(d.category == "absent").sum()} absent + '
          f'{(d.category == "inflated").sum()} inflated + '
          f'{(d.category == "annotation").sum()} annotation + '
          f'{(d.category == "unmeasurable").sum()} unmeasurable')


if __name__ == '__main__':
    main()
