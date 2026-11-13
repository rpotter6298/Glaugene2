"""Small-RNA class reference index: classify adapter-trimmed molecules by their first 20-mer."""
import gzip, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / 'refs'
K = 20

REFSEQ_RULES = [
    ('rRNA',    re.compile(r'RNA(5S|5-8S|18S|28S|45S)', re.I)),
    ('snRNA',   re.compile(r'\bRNU\d|small nuclear', re.I)),
    ('snoRNA',  re.compile(r'small nucleolar|SNOR|SCARNA', re.I)),
    ('YRNA',    re.compile(r'\bRNY\d|Ro60-associated', re.I)),
    ('SRP/7SK', re.compile(r'RN7SL|RN7SK|signal recognition', re.I)),
    ('vault',   re.compile(r'vault', re.I)),
]


def _fasta(path, want_human=True):
    op = gzip.open if str(path).endswith('.gz') else open
    name, seq = None, []
    with op(path, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                if name and seq:
                    yield name, ''.join(seq)
                hdr = line[1:]
                name = hdr if (not want_human or '_9606 ' in hdr or hdr.startswith('hsa')) else None
                seq = []
            elif name is not None:
                seq.append(line.upper().replace('U', 'T'))
    if name and seq:
        yield name, ''.join(seq)


def trna_name(header):
    """tRNA header -> 'Val-CAC' label; tolerates headers that omit the word 'anticodon'."""
    m = re.search(r'tRNA-([A-Za-z]+)\s*\((?:anticodon\s*)?([ACGTU]{3})\)', header, re.I)
    if m:
        return f'{m.group(1)}-{m.group(2).upper().replace("U", "T")}'
    m = re.search(r'tRNA-([A-Za-z]+)', header)
    return f'{m.group(1)}-???' if m else 'tRNA-unknown'


PRIORITY = ['tRNA', 'rRNA', 'snRNA', 'snoRNA', 'YRNA', 'SRP/7SK', 'vault',
            'miRNA', 'piRNA', 'other']


def sources(include_other=False):
    """[(class, label_fn, [(name, seq), ...]), ...], sorted into PRIORITY order."""
    out = []
    f = REFS / 'gtrnadb.fasta.gz'
    if f.exists():
        out.append(('tRNA', trna_name, list(_fasta(f))))
    mi = []
    for f in ['mature_v22.fa', 'mature_current.fa', 'hairpin_v22.fa']:
        p = REFS / f
        if p.exists():
            mi += [(n, s) for n, s in _fasta(p) if n.startswith('hsa')]
    if mi:
        out.append(('miRNA', lambda h: h.split()[0], mi))
    sn = []
    for f in ['snopy.fasta.gz', 'snodb.fasta.gz']:
        p = REFS / f
        if p.exists():
            sn += list(_fasta(p))
    if sn:
        out.append(('snoRNA', lambda h: h.split()[0], sn))
    p = REFS / 'pirbase.fasta.gz'
    if p.exists():
        out.append(('piRNA', lambda h: h.split()[0], list(_fasta(p))))
    p = REFS / 'refseq_human.fasta.gz'
    if p.exists():
        buckets = {}
        for n, s in _fasta(p):
            cls = 'other'
            for c, rx in REFSEQ_RULES:
                if rx.search(n):
                    cls = c; break
            buckets.setdefault(cls, []).append((n, s))
        wanted = ['rRNA', 'snRNA', 'snoRNA', 'YRNA', 'SRP/7SK', 'vault']
        if include_other:
            wanted.append('other')
        for c in wanted:
            if buckets.get(c):
                out.append((c, lambda h: h.split()[0], buckets[c]))
    # merge duplicate class entries (snoRNA arrives from two references) and order
    merged = {}
    for cls, fn, seqs in out:
        if cls in merged:
            merged[cls][1].extend(seqs)
        else:
            merged[cls] = [fn, list(seqs)]
    return [(c, merged[c][0], merged[c][1]) for c in PRIORITY if c in merged]


def build_index(verbose=True, include_other=False):
    """kmer -> (class, label, offset, source length). Higher-priority class wins."""
    idx = {}
    for cls, labeller, seqs in sources(include_other):
        new = taken = 0
        for hdr, seq in seqs:
            lab = labeller(hdr)
            for i in range(len(seq) - K + 1):
                km = seq[i:i + K]
                if km in idx:
                    taken += idx[km][0] != cls
                else:
                    idx[km] = (cls, lab, i, len(seq)); new += 1
        if verbose:
            print(f'  {cls:9s} {len(seqs):7,} seqs  ->  {new:8,} new {K}-mers'
                  + (f'  ({taken:,} already claimed by a higher class)' if taken else ''))
    if verbose:
        print(f'  index: {len(idx):,} {K}-mers')
    return idx
