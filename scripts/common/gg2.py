"""Shared loaders for GlauGene2. Everything reads from inside the repo."""
from pathlib import Path
import gzip, collections, os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

ARMS = {
    'BEA-raw':      ('counts/BEA-raw/counts_BEA-raw_main.tsv',       'v22'),
    'BEA-dedup':    ('counts/BEA-dedup/counts_BEA-dedup_main.tsv',   'v22'),
    'BEA-like':     ('counts/BEA-like/counts_BEA_main.tsv',          'v22'),
    'nfcore':       ('counts/nfcore/counts_nfcore_main.tsv',         'v23'),
    # UMI-dedup nfcore from raw FASTQs (nf-core ran with_umi=False)
    'nfcore-dedup': ('counts/nfcore-dedup/counts_nfcore-dedup_main.tsv', 'v23'),
    'CBGM-like':    ('counts/CBGM-like/counts_CBGM_main.tsv',        'v23'),
    # v22 re-run of CBGM-like, to match the comparator's miRBase release exactly
    'CBGM-v22':     ('counts/CBGM-v22/counts_CBGM-v22_main.tsv',    'v22'),
    # v22 re-run of PEXlens-like: all four Genes 2022 DE claims are v22 names
    'PEXlens-v22':  ('counts/PEXlens-v22/counts_PEXlens-v22_main.tsv', 'v22'),
    'PEXlens-like': ('counts/PEXlens-like/counts_PEXlens_main.tsv',  'v23'),
}
# Count matrix for this run; override with GG2_ARM=<name> to tile the pipeline.
PRIMARY = os.environ.get('GG2_ARM', 'BEA-dedup')   # supervisor's preferred arm
if PRIMARY not in ARMS:
    raise SystemExit(f'GG2_ARM={PRIMARY!r} is not one of: {", ".join(ARMS)}')

# SHARED: branch-independent FASTQ-derived outputs; PLOTS: this arm's outputs.
SHARED = ROOT / 'plots'
PLOTS = SHARED if PRIMARY == 'BEA-raw' else SHARED / 'branches' / PRIMARY

# QIAseq 3' adapter; insert is everything before it.
ADAPTER = 'AACTGTAGGCACCATCAAT'

# Claimed in counts/README.md (v22 names); EXCLUDE is the verified list.
CLAIMED_ARTIFACTS = ['hsa-miR-5701', 'hsa-miR-4502', 'hsa-miR-4791', 'hsa-miR-4454',
                     'hsa-miR-7704', 'hsa-miR-1246', 'hsa-miR-542-3p', 'hsa-miR-6785-5p']

# Stage 1 exclusion list, verified by scripts/stage_1/verify_suspects.py
# Sequence is absent from the library: the counts are alignment artifacts.
ARTIFACTS_ABSENT = [
    'hsa-miR-2682-5p',
    'hsa-miR-7704',
    'hsa-miR-4502',
    'hsa-miR-5701',      # -> SNORD43 (URS000022EFB6_9606)
    'hsa-miR-6785-5p',
    'hsa-miR-1303',
    'hsa-miR-542-3p',
    'hsa-miR-8054',
    'hsa-miR-4791',      # -> human snoRNA URS000053D4AB_9606
    'hsa-miR-1184',
    'hsa-miR-548v',
    'hsa-miR-4454',
    'hsa-miR-8089',
    'hsa-miR-657',
    'hsa-miR-4477b',
    'hsa-miR-4423-3p',
    'hsa-miR-4639-5p',
    'hsa-miR-6756-5p',
]

# Inflated: BEA assigns 5-380x more than strict branches; 0 standalone molecules.
ARTIFACTS_INFLATED = [
    'hsa-miR-3182',      # 5' half of tRNA-Val(CAC); see scripts/stage_1/case_study_mir3182.py
    'hsa-miR-4497',
    'hsa-miR-4448',
    'hsa-miR-5585-3p',
    'hsa-miR-4284',
    'hsa-miR-3609',
    'hsa-miR-935',
    'hsa-miR-1908-5p',
    'hsa-miR-4259',
]

# Present only inside longer transcripts; annotation error, not alignment error.
ARTIFACTS_ANNOTATION = ['hsa-miR-1246']

# 16 nt, unresolvable at read level; not excluded — flag any result involving it.
UNRESOLVED = ['hsa-miR-4485-5p']

EXCLUDE = ARTIFACTS_ABSENT + ARTIFACTS_INFLATED + ARTIFACTS_ANNOTATION

# Blind spots: zero in all 95 samples; drop these rows, never report as zero.
# Cause: featureCounts drops reads matching an rc-twin pair; 3 entries lack one.
BLIND_SPOTS = {
    'BEA-raw': ['hsa-miR-423-5p', 'hsa-miR-3184-3p', 'hsa-miR-103a-3p',
                'hsa-miR-103b-5p', 'hsa-miR-151b-3p', 'hsa-miR-3529-3p',
                'hsa-miR-7-5p', 'hsa-miR-1268a-5p'],
}
BLIND_SPOTS['BEA-dedup'] = BLIND_SPOTS['BEA-raw']


def rc(seq):
    return seq[::-1].translate(str.maketrans('ACGT', 'TGCA'))


def matrix(arm, drop_blind=True, drop_samples=True):
    """Counts for one branch, minus Stage 1 exclusions and this branch's blind spots."""
    m = drop_artifacts(counts(arm, drop_samples=drop_samples))
    if drop_blind:
        bad = set(BLIND_SPOTS.get(arm, []))
        if bad:
            M = name_map_v22_v23()
            bad |= {M[n][0] for n in bad if M.get(n, (None,))[0]}
            m = m.loc[[i for i in m.index if i not in bad]]
    return m


def drop_artifacts(m):
    """Remove the verified exclusion list from a count matrix (v22 or v23 names)."""
    M = name_map_v22_v23()
    bad = set(EXCLUDE)
    bad |= {M[n][0] for n in EXCLUDE if M.get(n, (None,))[0]}
    return m.loc[[i for i in m.index if i not in bad]]


def load_fasta(path, prefix='hsa'):
    d, n = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            n = line[1:].split()[0]
            d[n] = ''
        elif n and line:
            d[n] += line.replace('U', 'T')
    return {k: v for k, v in d.items() if k.startswith(prefix)}


def refs():
    """miRBase v22 (BEA's reference) and current/v23 human mature sequences."""
    r = ROOT / 'refs'
    return load_fasta(r / 'mature_v22.fa'), load_fasta(r / 'mature_current.fa')


# Whole-library exclusions; criterion is peer agreement, not read quality.
# Rationale: scripts/stage_1/diagnose_sample.py
EXCLUDE_SAMPLES = ['BEA25P077_RP_100_40A_G2']


def counts(arm, drop_samples=True):
    """Counts for one arm; drop_samples=False keeps EXCLUDE_SAMPLES (for QC figures)."""
    path, _ = ARMS[arm]
    m = pd.read_csv(ROOT / path, sep='\t', index_col=0)
    m.index = [str(i) for i in m.index]
    m = m[~m.index.isin(['nan'])]
    if drop_samples:
        m = m.drop(columns=[c for c in EXCLUDE_SAMPLES if c in m.columns])
    return m


SAMPLE_TABLES = ('patient_map.xlsx', 'patient_labels.xlsx')


def metadata():
    """Main-cohort sample table, from the first of SAMPLE_TABLES that exists; patient_labels.xlsx is the published subset, without the clinical columns."""
    import openpyxl
    src = next((ROOT / n for n in SAMPLE_TABLES if (ROOT / n).exists()), None)
    if src is None:
        raise SystemExit('no sample table at %s: expected one of %s'
                         % (ROOT, ', '.join(SAMPLE_TABLES)))
    wb = openpyxl.load_workbook(src, read_only=True)
    ws = wb['samples']
    rows = list(ws.iter_rows(values_only=True))
    hdr = list(rows[0])
    recs = [dict(zip(hdr, r)) for r in rows[1:] if any(x is not None for x in r)]
    df = pd.DataFrame([r for r in recs if r['cohort'] == 'main'])
    return df.set_index('sample')


def trim(read, min_adapter=6):
    """Insert before the 3' adapter; longest prefix flush at 3' end; None if no adapter visible."""
    p = read.find(ADAPTER)
    if p >= 0:
        return read[:p]
    for L in range(len(ADAPTER) - 1, min_adapter - 1, -1):
        q = read.find(ADAPTER[:L])
        if q > 0 and q + L == len(read):
            return read[:q]
    return None


def inserts(sample, n_reads=None, min_len=15):
    """Adapter-trimmed insert sequences from one main-cohort FASTQ."""
    path = ROOT / 'fastq' / 'main' / f'{sample}.fastq.gz'
    c = collections.Counter()
    with gzip.open(path, 'rt') as fh:
        for i, line in enumerate(fh):
            if n_reads and i >= n_reads * 4:
                break
            if i % 4 == 1:
                r = trim(line.strip())
                if r is not None and len(r) >= min_len:
                    c[r] += 1
    return c


def name_map_v22_v23():
    """v22 -> v23 names, sequence first (name-only is unsafe); returns {v22: (v23 or None, how)}."""
    v22, v23 = refs()
    by_seq = {}
    for n, s in v23.items():
        by_seq.setdefault(s, []).append(n)

    out = {}
    for n, s in v22.items():
        if s in by_seq:                                    # 1. exact sequence
            out[n] = (by_seq[s][0], 'seq')
            continue
        cands = [k for k in v23 if k == n or k.startswith(n + '-')]
        if cands:                                          # 2. best arm by overlap
            def overlap(k):
                t = v23[k]
                return max(i for i in range(min(len(s), len(t)) + 1)
                           if s[:i] == t[:i])
            best = max(cands, key=overlap)
            out[n] = (best, 'arm' if overlap(best) >= 15 else 'arm-weak')
            continue
        if n in v23:                                       # 3. exact name
            out[n] = (n, 'name')
            continue
        out[n] = (None, 'none')
    return out


def rna_input():
    """Qubit input-RNA (ng/ul) per sample, pre-library-prep; range truncated low by selection."""
    import openpyxl, re
    qc = ROOT / 'QC' / 'BEA25P077_RP.xlsx'
    if not qc.exists():
        md = metadata()
        if 'rna_ng_ul' not in md.columns:
            raise SystemExit('no input-RNA concentrations: expected %s or an rna_ng_ul '
                             'column in the sample table' % qc)
        return md.rna_ng_ul.astype(float).rename('rna_ng_ul')
    wb = openpyxl.load_workbook(qc, read_only=True)
    recs = {}
    for name in wb.sheetnames:
        ws = wb[name]
        rows = list(ws.iter_rows(values_only=True))
        hdr = [c for c in rows[0] if c]
        for r in rows[1:]:
            if r[0] is None:
                continue
            d = dict(zip(hdr, r))
            c = d.get('conc (ng/ul)') or d.get('Conc [ng/ul]')
            try:
                recs[str(d.get('ID'))] = float(c)
            except (TypeError, ValueError):
                pass
    md = metadata()
    tail = {re.sub(r'^BEA25P077_RP_\d+_', '', x): x for x in md.index}
    out = {tail[k]: v for k, v in recs.items() if k in tail}
    return pd.Series(out, name='rna_ng_ul')
