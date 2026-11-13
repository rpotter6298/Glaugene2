#!/usr/bin/env python3
"""Stage 6 — caller for the tissue x disease interaction model."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('GG2_ARM', 'BEA-dedup')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'stage_3'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gg2
import de
import mirna_level_enrichment as M
import pandas as pd

OUT = gg2.PLOTS / 'stage_6'
RSCRIPT = Path(__file__).resolve().parent / 'tissue_disease_interaction.R'
R_ENV = {**os.environ, 'R_PROFILE_USER': '/dev/null', 'R_LIBS': '/r_packages'}


def main():
    md, m = gg2.metadata(), gg2.matrix(gg2.PRIMARY)
    cols = M.paired_samples(md, m.columns)
    meta = md.loc[cols, ['patient', 'tissue_name', 'disease']].copy()
    npat = meta.groupby('patient').tissue_name.nunique()
    assert len(npat) == 42 and (npat == 2).all()
    print(f'paired libraries: {len(meta)} (42 patients)', flush=True)
    m = m[cols]
    sets = []
    for tis in ('lens', 'tear'):
        sub, _ = de.filtered(m, md, tis)
        sets.append(set(sub.index))
        print(f'  {tis}: {len(sub)} entries pass the filter on the paired set', flush=True)
    entries = [e for e in m.index if e in sets[0] | sets[1]]
    print(f'  union: {len(entries)} entries', flush=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        m.loc[entries, :].to_csv(td / 'c.tsv', sep='\t')
        meta.to_csv(td / 'm.tsv', sep='\t', index_label='sample')
        p = subprocess.run(
            ['Rscript', str(RSCRIPT), str(td / 'c.tsv'), str(td / 'm.tsv'),
             str(OUT / 'tissue_disease_interaction.tsv')],
            capture_output=True, text=True, env=R_ENV)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        print(p.stdout.rstrip(), flush=True)
    it = pd.read_csv(OUT / 'tissue_disease_interaction.tsv', sep='\t')
    print(f'  interaction: {int((it.inter_FDR < .05).sum())} at FDR<0.05, '
          f'{int((it.inter_p < .05).sum())} at raw P<0.05 '
          f'({.05 * len(it):.1f} expected) | tissue effect: '
          f'{int((it.tissue_FDR < .05).sum())} at FDR<0.05')
    print(f'wrote {OUT / "tissue_disease_interaction.tsv"}')


if __name__ == '__main__':
    main()
