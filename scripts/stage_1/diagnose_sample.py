#!/usr/bin/env python3
"""Is a sample technically defective, or merely unusual? Tissue percentiles on every technical measure we have.  usage: diagnose_sample.py [sample-id-fragment], default 40A"""
import os
import sys
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-dedup')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
import gg2
import de
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FRAG = sys.argv[1] if len(sys.argv) > 1 else '40A'


def pct(series, value):
    return float((series < value).mean() * 100)


def main():
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    hit = [c for c in m.columns if FRAG in c]
    assert len(hit) == 1, f'{FRAG} matched {hit}'
    S = hit[0]
    tis = md.loc[S, 'tissue_name']
    peers = [c for c in m.columns
             if c in md.index and md.loc[c, 'tissue_name'] == tis and c != S]
    print(f'{S}\n  tissue {tis} · {md.loc[S, "disease"]} · patient '
          f'{md.loc[S, "patient"]} · compared against the other {len(peers)} '
          f'{tis} samples\n')

    comp = pd.read_csv(gg2.SHARED / 'stage_1' / 'rna_composition.tsv',
                       sep='\t').set_index('sample')
    c = comp.loc[[S] + peers]
    rows = []

    def add(label, vals, fmt='{:.1%}'):
        v, p = vals[S], vals[peers]
        rows.append(dict(measure=label, value=fmt.format(v),
                         tissue_median=fmt.format(p.median()),
                         percentile=f'{pct(p, v):.0f}',
                         z=f'{(v - p.mean()) / p.std():+.1f}'))

    add('total reads', c.total_reads, '{:,.0f}')
    add('  too short', c.too_short / c.total_reads)
    add('  no adapter found', c.no_adapter / c.total_reads)
    add('  unclassified', c.unclassified / (c.classified + c.unclassified))
    for k in ['miRNA', 'tRNA', 'rRNA', 'snoRNA', 'snRNA', 'piRNA', 'YRNA']:
        add(f'  {k} (of classified)', c[k] / c.classified)

    sub, samples = de.filtered(m, md, tis)
    cpm = sub / sub.sum() * 1e6
    det = (sub > 0).sum()
    add('miRNAs detected', det.reindex([S] + peers), '{:.0f}')
    top5 = cpm.apply(lambda x: np.sort(x.values)[::-1][:5].sum() / 1e4)
    add('top-5 concentration', top5.reindex([S] + peers) / 100)

    lg = np.log2(cpm + 1)
    rho = {q: spearmanr(lg[q], lg[[r for r in samples if r != q]].mean(axis=1))[0]
           for q in samples}
    rho = pd.Series(rho)
    add('agreement with peers (Spearman)', rho, '{:.3f}')

    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print('\npercentile = position within this tissue (50 = typical);  '
          'z = standard deviations from the tissue mean')

    flag = [r['measure'].strip() for _, r in t.iterrows()
            if abs(float(r['z'])) >= 3]
    print('\nmeasures where the sample is >=3 SD from its tissue: '
          + (', '.join(flag) if flag else 'NONE'))


if __name__ == '__main__':
    main()
