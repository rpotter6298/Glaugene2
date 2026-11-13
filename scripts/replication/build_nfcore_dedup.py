#!/usr/bin/env python3
"""Build the nfcore-dedup arm as nf-core/smrnaseq 2.4.1 --with_umi would: UMICollapse on raw reads, counted via the mirtop export."""
import collections, gzip, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2, pandas as pd
from joblib import Parallel, delayed

# Parent of Glaugene2/ and of the NGI delivery P40752/; GG2_STUDY_ROOT overrides.
R = Path(os.environ.get('GG2_STUDY_ROOT',
                        Path(__file__).resolve().parents[3]))
QIA = 'AACTGTAGGCACCATCAAT'
IMG = 'quay.io/biocontainers/umicollapse:1.1.0--hdfd78af_0'
JAR = '/usr/local/share/umicollapse-1.1.0-0/umicollapse.jar'
EXPORT = R / 'P40752/smrnaseq-run/results/mirna_quant/mirtop/export'
OUT = gg2.ROOT / 'counts/nfcore-dedup/main'; FQDIR = gg2.ROOT / 'fastq/main'
meta = pd.read_csv(R / 'P40752/sample_metadata.tsv', sep='\t')
SAMPLES = [(a, FQDIR / f'{b}.fastq.gz') for a, b in zip(meta.ngi_id, meta.user_id)
           if (EXPORT / f'{a}_mirtop_rawData.tsv').exists()]
OUT.mkdir(parents=True, exist_ok=True); DEDUP = OUT / 'umicollapse'; DEDUP.mkdir(exist_ok=True)


def umicollapse(sid, fq, heap='11g'):
    """Run the pipeline's exact UMICollapse call; returns the deduplicated FASTQ path, or None.
    Single-threaded and memory-bound: the heap is the real limit."""
    out = DEDUP / f'{sid}.umi_dedup.fastq.gz'
    if out.exists() and out.stat().st_size > 1000:      # UMICollapse opens the gz (10 bytes) at start; a killed run leaves that behind
        return out
    fq = Path(os.path.realpath(fq))      # FASTQs are symlinks; docker must mount the real parent
    cmd = ['docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}',
           '-v', f'{fq.parent}:/in:ro', '-v', f'{DEDUP}:/out', IMG,
           'java', f'-Xmx{heap}', '-Xss999M', '-jar', JAR, 'fastq',
           '-i', f'/in/{fq.name}', '-o', f'/out/{out.name}', '--algo', 'dir', '--two-pass']
    with open(DEDUP / f'{sid}.umicollapse.log', 'w') as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not out.exists() or out.stat().st_size <= 1000:
        out.unlink(missing_ok=True); return None
    return out


def count(sid, fq, tag):
    raw = pd.read_csv(EXPORT / f'{sid}_mirtop_rawData.tsv', sep='\t')
    seq2mir = dict(zip(raw.seq, raw.mir)); wanted = set(seq2mir)
    c = collections.Counter()
    with gzip.open(fq, 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1: continue
            s = line.rstrip(); j = s.find(QIA)
            if j < 0: continue
            ins = s[:j]
            if ins in wanted: c[seq2mir[ins]] += 1
    pd.Series(c).sort_index().to_csv(OUT / f'{sid}.{tag}.tsv', sep='\t', header=False)
    return sum(c.values())


def one(sid, fq, heap='11g'):
    if (OUT / f'{sid}.dedup.tsv').exists():
        return f'{sid}: cached'
    n_reads = count(sid, fq, 'reads') if not (OUT / f'{sid}.reads.tsv').exists() else None
    ded = umicollapse(sid, fq, heap)
    if ded is None:
        return f'{sid}: FAILED (heap {heap})'
    n_ded = count(sid, ded, 'dedup')
    return f'{sid}: reads {n_reads}  dedup {n_ded:,}' + (f'  ({n_ded/n_reads:.3f})' if n_reads else '')


def assemble():
    """Per-sample .dedup.tsv -> counts_nfcore-dedup/counts_nfcore-dedup_main.tsv (user_id columns);
    the previous exact-(insert,UMI) matrix is kept as counts_nfcore-dedup-exact_main.tsv."""
    meta = pd.read_csv(R / 'P40752/sample_metadata.tsv', sep='\t'); n2u = dict(zip(meta.ngi_id, meta.user_id))
    cols = {n2u[f.name.split('.')[0]]: pd.read_csv(f, sep='\t', header=None, index_col=0)[1]
            for f in sorted(OUT.glob('*.dedup.tsv')) if f.name.split('.')[0] in n2u}
    mat = pd.DataFrame(cols).fillna(0).astype(int).sort_index(); mat.index.name = 'miRNA'
    final = gg2.ROOT / 'counts/nfcore-dedup/counts_nfcore-dedup_main.tsv'
    legacy = gg2.ROOT / 'counts/nfcore-dedup/counts_nfcore-dedup-exact_main.tsv'
    if final.exists() and not legacy.exists():
        old = pd.read_csv(final, sep='\t', index_col=0)
        if set(old.columns) == set(mat.columns) and not old.equals(mat): old.to_csv(legacy, sep='\t')
    mat.to_csv(final, sep='\t')
    print(f'assembled {mat.shape[0]} miRNAs x {mat.shape[1]} samples -> {final}; library size median {mat.sum().median():,.0f}')


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5     # five 11 GB JVMs (~35 GB typical) on a 61 GB box
    print(f'{len(SAMPLES)} samples, {n} parallel', flush=True)
    res = Parallel(n_jobs=n, verbose=0, return_as='generator')(delayed(one)(a, b) for a, b in SAMPLES)
    failed = []
    for r, (sid, fq) in zip(res, SAMPLES):
        print(r, flush=True)
        if 'FAILED' in r: failed.append((sid, fq))
    for sid, fq in failed:
        print(one(sid, fq, heap='40g'), flush=True)
    missing = [sid for sid, _ in SAMPLES if not (OUT / f'{sid}.dedup.tsv').exists()]
    if missing:
        print('STILL MISSING:', missing); sys.exit(1)
    assemble()


if __name__ == '__main__':
    main()
