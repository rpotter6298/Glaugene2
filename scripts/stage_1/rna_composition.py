#!/usr/bin/env python3
"""What RNA classes actually make up each library: adapter-trimmed reads, first 20-mer looked up in the rnaclass.py index; supersedes trna_quant.py. A FLOOR on every class, not a partition of the library."""
import sys, gzip, csv
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2, rnaclass

OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
K, NPROC = rnaclass.K, 12
CLASSES = ['miRNA', 'tRNA', 'rRNA', 'snoRNA', 'snRNA', 'piRNA', 'YRNA',
           'SRP/7SK', 'vault']
IDX = None


def _init():
    global IDX
    IDX = rnaclass.build_index(verbose=False)


def classify(sample):
    total = short = no_adapter = 0
    cls = {c: 0 for c in CLASSES}
    trna = {}
    frag = {"5'": 0, "3'": 0, 'internal': 0}
    with gzip.open(gg2.ROOT / 'fastq' / 'main' / f'{sample}.fastq.gz', 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            total += 1
            mol = gg2.trim(line.strip())
            if mol is None:
                no_adapter += 1; continue
            if len(mol) < K:
                short += 1; continue
            hit = IDX.get(mol[:K])
            if not hit:
                continue
            c, lab, off, ln = hit
            if c in cls:
                cls[c] += 1
            if c == 'tRNA':
                trna[lab] = trna.get(lab, 0) + 1
                if off == 0:
                    frag["5'"] += 1
                elif off + len(mol) >= ln - 2:
                    frag["3'"] += 1
                else:
                    frag['internal'] += 1
    classified = sum(cls.values())
    return (sample, total, short, no_adapter, classified, cls, trna, frag)


def main():
    md = gg2.metadata()
    samples = list(md.index)
    print(f'classifying {len(samples)} samples into {len(CLASSES)} RNA classes, '
          f'{NPROC} workers')
    rows, comp = [], []
    with Pool(NPROC, initializer=_init) as pool:
        for n, (s, tot, sh, na, clf, cls, trna, frag) in enumerate(
                pool.imap_unordered(classify, samples), 1):
            r = dict(sample=s, tissue=md.loc[s, 'tissue_name'],
                     disease=md.loc[s, 'disease'], total_reads=tot,
                     too_short=sh, no_adapter=na, classified=clf,
                     unclassified=tot - sh - na - clf)
            for c in CLASSES:
                r[c.replace('/', '_')] = cls[c]
            r.update(trna5=frag["5'"], trna3=frag["3'"], trna_int=frag['internal'])
            rows.append(r)
            for k, v in trna.items():
                comp.append(dict(sample=s, trna=k, reads=v))
            print(f'  [{n}/{len(samples)}] {s}  miRNA {100*cls["miRNA"]/tot:5.2f}%  '
                  f'tRNA {100*cls["tRNA"]/tot:5.2f}%  rRNA {100*cls["rRNA"]/tot:5.2f}%',
                  flush=True)
    with open(OUT / 'rna_composition.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r['sample']))
    with open(OUT / 'trna_composition.tsv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['sample', 'trna', 'reads'], delimiter='\t')
        w.writeheader(); w.writerows(comp)
    print(f'\nwrote {OUT}/rna_composition.tsv and trna_composition.tsv')


if __name__ == '__main__':
    main()
