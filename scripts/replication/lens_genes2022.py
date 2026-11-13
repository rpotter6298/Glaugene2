#!/usr/bin/env python3
"""Replication B — Genes 2022 lens capsule (PEXlens-v22 arm)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'PEXlens-v22')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2
import numpy as np
import pandas as pd

OUT = gg2.PLOTS / 'replication'
OUT.mkdir(parents=True, exist_ok=True)
R_ENV = {**os.environ, 'R_PROFILE_USER': '/dev/null', 'R_LIBS': '/r_packages'}
CPM_MIN, FRAC = 3.7, 0.50
ALPHA = 0.10


def main():
    md = gg2.metadata()
    m = gg2.counts(gg2.PRIMARY)                      # as-published: no exclusions
    lens = [c for c in m.columns if md.loc[c, 'tissue_name'] == 'lens']
    t = m[lens]
    meta = md.loc[lens, ['sex', 'disease']].copy()
    meta['sex'] = meta['sex'].astype(str)
    n_d = int((meta.disease == 'diseased').sum())
    n_c = int((meta.disease == 'control').sum())
    print(f'{gg2.PRIMARY}: lens n={len(lens)} ({n_d} XFG / {n_c} control)   '
          f'theirs: 5 PEX / 5 control\n')

    # ---- their Table 2: composition by group ----
    theirs2 = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_top10.tsv', sep='\t')
    rows = []
    for grp, lab in [('control', 'control'), ('diseased', 'pex')]:
        cols = [c for c in lens if md.loc[c, 'disease'] == grp]
        share = t[cols].sum(axis=1) / t[cols].sum().sum() * 100
        for e in theirs2.miRNA:
            rows.append(dict(group=lab, miRNA=e,
                             ours_pct=round(float(share.get(e, 0.0)), 2)))
    comp = pd.DataFrame(rows).pivot(index='miRNA', columns='group', values='ours_pct')
    comp = theirs2.set_index('miRNA').join(comp)
    comp.to_csv(OUT / 'lens_genes2022_composition.tsv', sep='\t')
    print('=== their Table 2 (top 10 abundant) vs ours, % of reads ===')
    print(f'{"miRNA":18s}{"ctrl thm":>9}{"ctrl us":>9}   {"PEX thm":>9}{"XFG us":>9}')
    for e, r in comp.iterrows():
        print(f'{e.replace("hsa-",""):18s}{r.control_pct:>9.1f}{r["control"]:>9.2f}   '
              f'{r.pex_pct:>9.1f}{r["pex"]:>9.2f}')
    ctl = [c for c in lens if md.loc[c, 'disease'] == 'control']
    sh = (t[ctl].sum(axis=1) / t[ctl].sum().sum() * 100).sort_values(ascending=False)
    print(f'\ntop-5 concentration : theirs 90%   ours {sh.head(5).sum():.1f}%')
    print(f'top-10 concentration: theirs ~95%  ours {sh.head(10).sum():.1f}%')

    # ---- their filter, then TMM + edgeR LRT ----
    cpm = t / t.sum() * 1e6
    keep = (cpm >= CPM_MIN).sum(axis=1) >= FRAC * len(lens)
    sub = t[keep]
    print(f'\ntheir filter ({CPM_MIN} CPM in >={FRAC:.0%} of samples): '
          f'{len(sub)}/{len(t)} miRNAs retained')

    with tempfile.TemporaryDirectory() as td:
        cf, mf, of = f'{td}/c.tsv', f'{td}/m.tsv', f'{td}/o.tsv'
        sub.to_csv(cf, sep='\t')
        meta.to_csv(mf, sep='\t')
        r = subprocess.run(['Rscript', str(Path(__file__).with_suffix('.R')),
                            cf, mf, of], env=R_ENV, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout, r.stderr); raise SystemExit('edgeR failed')
        print(r.stdout.strip())
        de = pd.read_csv(of, sep='\t')

    de = de.rename(columns={'PValue': 'p'}).set_index('entry')
    de.to_csv(OUT / 'lens_genes2022_de.tsv', sep='\t')

    # ---- their Table 3: the six they report ----
    theirs3 = pd.read_csv(gg2.ROOT / 'refs' / 'genes2022_lens_de.tsv', sep='\t')
    print(f'\n=== their Table 3 vs ours (their method, their threshold p<{ALPHA}) ===')
    print(f'{"miRNA":18s}{"their logFC":>12}{"their p":>10}   '
          f'{"our logFC":>10}{"our p":>9}  {"":>3}')
    hit = 0
    for _, r in theirs3.iterrows():
        if r.miRNA not in de.index:
            state = 'below their expression filter in our data'
            print(f'{r.miRNA.replace("hsa-",""):18s}{r.logFC:>12.2f}{r.p:>10.4f}   '
                  f'{state:>25s}')
            continue
        o = de.loc[r.miRNA]
        same_dir = np.sign(o.logFC) == np.sign(r.logFC)
        flag = ''
        if o.p < ALPHA and same_dir:
            flag = 'REPLICATED'; hit += 1
        elif o.p < ALPHA:
            flag = 'sig, WRONG DIRECTION'
        print(f'{r.miRNA.replace("hsa-",""):18s}{r.logFC:>12.2f}{r.p:>10.4f}   '
              f'{o.logFC:>10.2f}{o.p:>9.4f}  {flag}')
    print(f'\nreplicated at their own threshold and direction: {hit}/{len(theirs3)}')

    n_sig = int((de.p < ALPHA).sum())
    n_crit = int(((de.p < ALPHA) & (de.logFC.abs() >= 1.5)).sum())
    print(f'\nour whole-lens result under their method: {n_sig}/{len(de)} at p<{ALPHA}, '
          f'{n_crit} of those also |logFC|>=1.5 (their headline criterion; theirs: 4)')
    print(f'top by p: ')
    for e, r in de.nsmallest(6, 'p').iterrows():
        print(f'   {e.replace("hsa-",""):18s} logFC {r.logFC:+.2f}  p={r.p:.4f}  '
              f'FDR={r.FDR:.3f}')
    print(f'\nwrote {OUT}/lens_genes2022_de.tsv and lens_genes2022_composition.tsv')


if __name__ == '__main__':
    main()
