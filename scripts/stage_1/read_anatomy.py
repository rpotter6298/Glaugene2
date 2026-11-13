#!/usr/bin/env python3
"""What a QIAseq small-RNA read actually contains, measured not assumed: per-cycle entropy separates biology (~1.5), fixed adapter (0) and random UMI (~2); the two-molecule comparison is in case_study_mir5701.py."""
import sys, gzip, collections, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
SAMPLE = 'BEA25P077_RP_100_40A_G2'
READ_LEN, N_SAMPLE = 72, 60000
C_RNA, C_ADP, C_UMI, C_ILL = '#3a6ea5', '#b9b9b9', '#a98fc4', '#7d7d7d'
C_GOLD, C_RED = '#b8860b', '#c1443f'


def collect():
    """Reads with a 22 nt insert (in phase), plus one example of each kind."""
    adp = gg2.ADAPTER
    inphase, ctrl, flag = [], None, None
    core = 'ACGGGCGGACAGAAACTGTGTGCTGATTGT'
    with gzip.open(gg2.ROOT / 'fastq' / 'main' / f'{SAMPLE}.fastq.gz', 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            s = line.strip()
            if s[22:22 + len(adp)] == adp:
                if len(inphase) < N_SAMPLE:
                    inphase.append(s)
            if ctrl is None and s.startswith('TGGACGGAGAACTGATAAGGGT'):
                ctrl = s
            if flag is None and core in s:
                flag = s
            if len(inphase) >= N_SAMPLE and ctrl and flag:
                break
    return inphase, ctrl, flag


def entropy_profile(reads):
    H = []
    for p in range(READ_LEN):
        c = collections.Counter(r[p] for r in reads if len(r) > p)
        n = sum(c[b] for b in 'ACGT')
        if not n:
            H.append(np.nan); continue
        f = [c[b] / n for b in 'ACGT']
        H.append(-sum(x * math.log2(x) for x in f if x > 0))
    return np.array(H)


def segs_for(ins_len, read_len=READ_LEN):
    out = [(0, min(ins_len, read_len), C_RNA, 'the RNA molecule')]
    a = min(read_len, ins_len + len(gg2.ADAPTER))
    if a > ins_len:
        out.append((ins_len, a, C_ADP, 'QIAseq adapter'))
    if a < read_len:
        u = min(read_len, a + 12)
        out.append((a, u, C_UMI, 'UMI'))
        if u < read_len:
            out.append((u, read_len, C_ILL, 'Illumina adapter'))
    return out


def main():
    inphase, ctrl, flag = collect()
    H = entropy_profile(inphase)
    print(f'{len(inphase):,} in-phase reads')

    fig = plt.figure(figsize=(15, 6.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.10, 1.25], hspace=.40,
                          left=.07, right=.965, top=.865, bottom=.085)
    fig.suptitle('QIAseq miRNA Read Structure', fontsize=17, weight='bold', y=.968)
    fig.text(.5, .918, f'72-cycle single-end read  ·  sample {SAMPLE}',
             ha='center', fontsize=10.5, color='#555')

    # Panel A: the molecule, and what each part is for
    ax = fig.add_subplot(gs[0]); ax.axis('off')
    ax.set_xlim(0, 100); ax.set_ylim(0, 1)
    ax.text(0, 1.03, 'The molecule that goes into the sequencer — and what each part does',
            fontsize=12.5, weight='bold', va='top')
    blocks = [
        (0, 9, '#dddddd', 'P5',
         'sticks to the\nflow cell; lets the\nmolecule be copied\ninto a bright cluster'),
        (9, 20, '#eeeeee', 'Read-1\nprimer site',
         'the sequencing primer\nlands here — cycle 1 is\nthe next base along'),
        (20, 38, C_RNA, 'the RNA',
         'your data.\nsequenced FIRST,\nso its length decides\nwhat else fits'),
        (38, 55, C_ADP, 'QIAseq adapter',
         'glued to the RNA 3\' end\nin the lab; primes the\nRNA→DNA copy step;\nmarks where RNA ends'),
        (55, 66, C_UMI, 'UMI',
         'random 12 nt tag, one\nper ORIGINAL molecule —\nlets PCR duplicates\nbe collapsed'),
        (66, 100, C_ILL, 'Illumina adapter + index + P7',
         'read-2 primer site; the sample barcode (read separately,\n'
         'which is why it shows in the FASTQ header not the sequence);\n'
         'P7 = the other flow-cell anchor'),
    ]
    for a, b, c, lb, role in blocks:
        ax.add_patch(Rectangle((a, .50), b - a, .22, facecolor=c, edgecolor='#555', lw=.9))
        ax.text((a + b) / 2, .61, lb, fontsize=9.2, ha='center', va='center',
                color='w' if c in (C_RNA, C_ILL) else '#222', weight='bold')
        ax.text((a + b) / 2, .42, role, fontsize=8.0, ha='center', va='top',
                color='#444', linespacing=1.45)
    ax.annotate('', xy=(97, .82), xytext=(20, .82),
                arrowprops=dict(arrowstyle='-|>', color=C_RED, lw=2.4))
    ax.text(20, .845, 'Read 1: 72 cycles, starting at the RNA — not at the UMI',
            fontsize=11, color=C_RED, weight='bold', va='bottom')
    # Panel B: measured entropy per cycle
    ax = fig.add_subplot(gs[1])
    for a, b, c, lb in segs_for(22):
        ax.axvspan(a - .5, b - .5, color=c, alpha=.28, zorder=0)
        ax.text((a + b - 1) / 2, 2.24, lb, fontsize=9.5, ha='center', weight='bold',
                color='#333')
    ax.plot(range(READ_LEN), H, color='#222', lw=1.9, marker='o', ms=2.6)
    ax.axhline(2, color=C_UMI, ls=':', lw=1.4)
    ax.text(READ_LEN - .5, 2.02, 'max randomness (2.0) ', fontsize=8.5, ha='right',
            color='#6b4e92')
    ax.set_ylim(-.15, 2.5); ax.set_xlim(-1, READ_LEN)
    ax.set_ylabel('base variability\n(Shannon entropy)', fontsize=10)
    ax.set_xlabel('sequencing cycle', fontsize=10)
    ax.set_title(f'Entropy Graph — {len(inphase):,} reads with a 22 nt insert',
                 fontsize=12.5, weight='bold', pad=26)
    ax.grid(alpha=.25); ax.set_axisbelow(True)
    for x, t, col in [(31, 'ALWAYS the same base\n= a fixed adapter', '#444'),
                      (46.5, 'PERFECTLY random\n= the UMI', '#6b4e92'),
                      (10, 'variable\n= real biology', '#28527d')]:
        ax.annotate(t, xy=(x, H[int(x)]), xytext=(x, 1.05 if col == '#444' else 0.62),
                    fontsize=9, ha='center', color=col, weight='bold',
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.2))

    fig.savefig(OUT / 'fig_read_anatomy.png', dpi=150)
    print(f'wrote {OUT}/fig_read_anatomy.png')


if __name__ == '__main__':
    main()
