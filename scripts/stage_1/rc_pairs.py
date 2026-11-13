#!/usr/bin/env python3
"""Reverse-complement miRNA pairs (one molecule, two entry names), and how differently each branch handles the ambiguous reads."""
import sys, gzip
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

# gg2.counts() not gg2.matrix(): this figure is ABOUT the excluded entries
OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
PAIR = ('hsa-miR-423-5p', 'hsa-miR-3184-3p')
SAMPLE = 'BEA25P077_RP_100_40A_G2'
# red: one molecule counted twice is unambiguously wrong; amber: discarding the read is lossy but defensible
C_DISCARD, C_ONE, C_DOUBLE = '#b8860b', '#2e8b57', '#c1443f'
C_A, C_B = '#3a6ea5', '#8f6bb0'


def rc(s):
    return s[::-1].translate(str.maketrans('ACGT', 'TGCA'))


def rc_twins(v22):
    by = {}
    for n, s in v22.items():
        by.setdefault(s, []).append(n)
    out = []
    for n, s in v22.items():
        for t in by.get(rc(s), []):
            if n < t:
                out.append((n, t))
    return out


def example_read(seq):
    with gzip.open(gg2.ROOT / 'fastq' / 'main' / f'{SAMPLE}.fastq.gz', 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1 and seq in line:
                return line.strip()
    return None


def branch_values(a, b):
    M = gg2.name_map_v22_v23()
    rows = {}
    for arm, (_, ver) in gg2.ARMS.items():
        m = gg2.counts(arm); tot = m.sum()
        vals = []
        for n in (a, b):
            k = n if n in m.index else (M.get(n, (None,))[0] if ver == 'v23' else None)
            vals.append(float((m.loc[k] / tot * 1e6).mean()) if k and k in m.index else 0.0)
        rows[arm] = vals
    return rows


def main():
    v22, _ = gg2.refs()
    a, b = PAIR
    A, B = v22[a], v22[b]
    twins = rc_twins(v22)
    bea = gg2.counts('BEA-raw')
    zero_twins = sum(1 for x, y in twins
                     for n in (x, y) if n in bea.index and bea.loc[n].sum() == 0)
    n_twin_entries = len({n for p in twins for n in p if n in bea.index})
    read = example_read(A)
    vals = branch_values(a, b)
    print(f'{len(twins)} rc pairs, {n_twin_entries} entries; {zero_twins} zero in BEA-raw')

    fig = plt.figure(figsize=(14.5, 10.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.02, 0.92, 1.15], hspace=.42,
                          left=.055, right=.97, top=.885, bottom=.06)
    fig.suptitle('Reverse-complement pairs', fontsize=17, weight='bold', y=.968)
    fig.text(.5, .922, 'miRBase annotates some loci on both strands — one molecule, '
                       'two entry names. The branches disagree about what to count.',
             ha='center', va='center', fontsize=10.5, color='#555')

    # ---- Panel 1: the sequences ----
    ax = fig.add_subplot(gs[0]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, 'The two entries are exact reverse complements',
            fontsize=12.5, weight='bold', va='top')
    steps = [(a, A, C_A), ('reversed', A[::-1], '#999'),
             ('complemented', rc(A), C_B), (b, B, C_B)]
    y = .70
    for lab, seq, col in steps:
        ax.text(.005, y, lab, fontsize=10, va='center', ha='left',
                weight='bold' if lab in (a, b) else 'normal',
                color=col if lab in (a, b) else '#666')
        ax.text(.235, y, seq, family='monospace', fontsize=12, va='center', color=col)
        y -= .215
    ax.annotate('', xy=(.565, .055), xytext=(.565, .275),
                arrowprops=dict(arrowstyle='-', color=C_B, lw=1.6))
    ax.plot([.555, .565], [.055, .055], color=C_B, lw=1.6)
    ax.plot([.555, .565], [.275, .275], color=C_B, lw=1.6)
    ax.text(.578, .165, 'identical', fontsize=11.5, weight='bold', color=C_B, va='center')

    # ---- Panel 2: one read matches both ----
    ax = fig.add_subplot(gs[1]); ax.axis('off')
    ax.set_xlim(-2, len(read) + 2); ax.set_ylim(0, 1)
    ax.text(-2, 1.02, 'One read matches both entries', fontsize=12.5,
            weight='bold', va='top')
    i = read.find(A)
    for yy, seq, lab, col, note in [
            (.60, read, 'read', '#333', ''),
            (.24, rc(read), 'reverse complement of the same read', '#333', '')]:
        hi = (i, i + len(A)) if yy > .4 else (len(read) - i - len(A), len(read) - i)
        ax.add_patch(Rectangle((hi[0] - .5, yy - .075), hi[1] - hi[0], .15,
                               facecolor='#fbe6c8' if yy > .4 else '#e7defa',
                               edgecolor=C_A if yy > .4 else C_B, lw=1.6, zorder=1))
        for j, base in enumerate(seq):
            inhi = hi[0] <= j < hi[1]
            ax.text(j, yy, base, family='monospace', fontsize=7.6, ha='center',
                    va='center', zorder=3, weight='bold' if inhi else 'normal',
                    color=(C_A if yy > .4 else C_B) if inhi else '#555')
        ax.text(-2, yy - .155, lab, fontsize=9.5, color='#666', va='top')
        ax.text((hi[0] + hi[1]) / 2 - .5, yy + .115,
                a if yy > .4 else b, fontsize=10, weight='bold', ha='center',
                color=C_A if yy > .4 else C_B)

    # ---- Panel 3: what each branch does ----
    ax = fig.add_subplot(gs[2])
    arms = list(gg2.ARMS)
    x = np.arange(len(arms)); w = .36
    va = [vals[m][0] for m in arms]; vb = [vals[m][1] for m in arms]
    ax.bar(x - w/2, va, w, color=C_A, label=a, edgecolor='#555', linewidth=.6)
    ax.bar(x + w/2, vb, w, color=C_B, label=b, edgecolor='#555', linewidth=.6)
    ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=10)
    ax.set_ylabel('mean CPM', fontsize=10)
    ax.legend(fontsize=9.5, frameon=False, loc='upper left')
    ax.grid(axis='y', alpha=.25); ax.set_axisbelow(True)
    top = max(max(va), max(vb))
    ax.set_ylim(0, top * 1.34)
    for i_, m in enumerate(arms):
        p, q = vals[m]
        if max(p, q) < 1:
            txt, col = 'Read Discarded', C_DISCARD
        elif min(p, q) > 1 and abs(p - q) / max(p, q) < .05:
            txt, col = 'Both Counted', C_DOUBLE
        else:
            txt, col = 'One Counted', C_ONE
        ax.text(i_, max(p, q) + top * .05, txt, ha='center', fontsize=10,
                weight='bold', color=col)
    ax.set_title(f'{n_twin_entries} entries have a reverse-complement twin — '
                 f'all {zero_twins} are zero in BEA-raw (0 of 2,315 non-zero entries are)',
                 fontsize=11.5, weight='bold', pad=10)
    fig.savefig(OUT / 'fig_rc_pairs.png', dpi=150)
    print(f'wrote {OUT}/fig_rc_pairs.png')


if __name__ == '__main__':
    main()
