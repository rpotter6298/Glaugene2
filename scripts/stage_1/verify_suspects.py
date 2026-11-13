#!/usr/bin/env python3
"""Read-level verification of every suspected artifact: full reads, no adapter trimming (method from case_study_mir5701.py); real miRNAs land at observed/assigned ~0.3-0.8."""
import sys, gzip, pickle, collections
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

# gg2.counts() not gg2.matrix(): verification needs the unfiltered matrix

OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / '.insert_cache'; CACHE.mkdir(exist_ok=True)
ILLUMINA = 'AGATCGGAAGAGC'
N_EXACT, N_SPECTRUM = 4, 2          # samples used for each test
CONTROLS = ['hsa-miR-184', 'hsa-let-7a-5p', 'hsa-miR-21-5p', 'hsa-miR-375-3p',
            'hsa-miR-200c-3p', 'hsa-let-7c-5p']


def bea_reads(sample):
    """Reads as BEA had them: Illumina adapter stripped only."""
    cf = CACHE / f'{sample}.bea.pkl'
    if cf.exists():
        return pickle.loads(cf.read_bytes())
    c = collections.Counter()
    with gzip.open(gg2.ROOT / 'fastq' / 'main' / f'{sample}.fastq.gz', 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                s = line.strip(); p = s.find(ILLUMINA)
                c[s[:p] if p > 0 else s] += 1
    cf.write_bytes(pickle.dumps(c)); return c


def candidates():
    """Flagged entries plus the cross-branch scan."""
    v22, _ = gg2.refs(); M = gg2.name_map_v22_v23()
    def cpm(a):
        m = gg2.counts(a); return (m / m.sum() * 1e6).mean(axis=1)
    bea, nf, px = cpm('BEA-raw'), cpm('nfcore'), cpm('PEXlens-like')
    out = {}
    for n, c in bea.items():
        if n not in v22:
            continue
        alt, how = M.get(n, (None, 'none'))
        flagged = n in gg2.CLAIMED_ARTIFACTS
        if flagged:
            out[n] = ('flagged', c, how)
        elif c >= 10 and alt and max(nf.get(alt, 0), px.get(alt, 0)) < 1.0:
            out[n] = ('scan', c, how)
    return out


def identify(seq, pooled):
    """What is the dominant read carrying this window?"""
    v22, v23 = gg2.refs()
    for db in ['snopy', 'snodb']:
        f = gg2.ROOT / 'refs' / f'{db}.fasta.gz'
        if not f.exists():
            continue
        name, cur = None, ''
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('>'):
                    if name and '9606' in name and seq in cur:
                        return name.split()[0] + ' (human snoRNA)'
                    name, cur = line[1:], ''
                else:
                    cur += line.upper().replace('U', 'T')
        if name and '9606' in name and seq in cur:
            return name.split()[0] + ' (human snoRNA)'
    for n, s in v23.items():
        if seq == s:
            return n + ' (miRBase v23)'
    return None


def main():
    v22, _ = gg2.refs()
    md = gg2.metadata(); bea = gg2.counts('BEA-raw')
    cands = candidates()
    panel = list(cands) + [c for c in CONTROLS if c in v22]
    print(f'{len(cands)} candidates '
          f'({sum(1 for v in cands.values() if v[0]=="flagged")} flagged, '
          f'{sum(1 for v in cands.values() if v[0]=="scan")} from scan)'
          f' + {len(panel)-len(cands)} controls')

    cpm = bea / bea.sum() * 1e6
    rank = cpm.loc[[c for c in cands if c in cpm.index]].sum(axis=0).sort_values(ascending=False)
    samples = list(rank.index[:N_EXACT])
    pooled = collections.Counter()
    for s in samples:
        print(f'  reading {s} ...', flush=True)
        pooled.update(bea_reads(s))
    print(f'  pooled {sum(pooled.values()):,} reads, {len(pooled):,} unique')
    mols = collections.Counter()          # trim once, reuse for every entry
    for k, v in pooled.items():
        t = gg2.trim(k)
        if t:
            mols[t] += v
    print(f'  {len(mols):,} distinct molecules after adapter trimming')

    # numpy matrix for local-alignment spectra (fewer samples, memory)
    spec_samples = samples[:N_SPECTRUM]
    sub = collections.Counter()
    for s in spec_samples:
        sub.update(bea_reads(s))
    keys = list(sub); L = 72
    arr = np.full((len(keys), L), 255, np.uint8)
    for i, k in enumerate(keys):
        b = np.frombuffer(k[:L].encode(), np.uint8); arr[i, :len(b)] = b
    w = np.array([sub[k] for k in keys])

    rows = []
    for name in panel:
        seq = v22[name]
        assigned = int(bea.loc[name, samples].sum()) if name in bea.index else np.nan
        exact = mols.get(seq, 0)
        contained = sum(v for k, v in pooled.items() if seq in k)
        r = np.frombuffer(seq.encode(), np.uint8); k = len(r)
        best = np.zeros(len(keys), np.int16)
        for s0 in range(L - k + 1):
            np.maximum(best, (arr[:, s0:s0 + k] == r).sum(1).astype(np.int16), out=best)
        near = int(w[best == k - 1].sum())
        top = ''
        if best.max() >= k - 1:
            idx = np.where(best >= k - 1)[0]
            j = idx[np.argmax(w[idx])]
            ins = keys[j]
            p = ins.find(gg2.ADAPTER)
            if p < 0:
                for Ln in range(len(gg2.ADAPTER) - 1, 5, -1):
                    q = ins.find(gg2.ADAPTER[:Ln])
                    if q > 0 and q + Ln == len(ins):
                        p = q; break
            mol = ins[:p] if p > 0 else ins
            who = identify(mol, pooled)
            top = f'{len(mol)}nt' + (f': {who}' if who else '')
        kind = cands[name][0] if name in cands else 'control'
        ratio = contained / assigned if assigned else np.nan
        rows.append(dict(entry=name, kind=kind, nt=len(seq), assigned=assigned,
                         exact=exact, contained=contained, ratio=ratio,
                         near_miss=near, source=top))
        print(f'  {name:20s} {kind:8s} assigned={assigned:>7} exact={exact:>7} '
              f'ratio={ratio:.3f}' if assigned else f'  {name:20s} {kind}')

    d = pd.DataFrame(rows)
    lo = d[d.kind == 'control'].ratio.min()
    d['verdict'] = np.where(d.kind == 'control', 'control',
                   np.where(d.nt < 18, 'TOO SHORT to test',
                   np.where(d.ratio.fillna(0) < 0.02, 'ARTIFACT',
                   np.where(d.ratio >= lo * .5, 'sequence present', 'equivocal'))))
    d.sort_values(['kind', 'assigned'], ascending=[True, False]).to_csv(
        OUT / 'suspects_verified.tsv', sep='\t', index=False)
    print(f'\ncontrol band: observed/assigned {lo:.2f}-{d[d.kind=="control"].ratio.max():.2f}')
    print(d.sort_values(['kind', 'assigned'], ascending=[True, False])
           [['entry', 'kind', 'nt', 'assigned', 'exact', 'ratio', 'verdict', 'source']]
           .to_string(index=False))
    print(f'\nwrote {OUT}/suspects_verified.tsv')


if __name__ == '__main__':
    main()
