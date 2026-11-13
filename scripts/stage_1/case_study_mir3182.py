#!/usr/bin/env python3
"""Case study: hsa-miR-3182 is the 5' half of tRNA-Val(CAC), the largest artifact in the matrix (4.9% of BEA-raw reads); same method as case_study_mir5701.py."""
import sys, gzip, pickle, collections
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

# gg2.counts() not gg2.matrix(): the case study is ABOUT an excluded entry
OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / '.insert_cache'
SAMPLE = 'BEA25P077_RP_100_40A_G2'
ILLUMINA, UMI_LEN = 'AGATCGGAAGAGC', 12
ENTRY, CONTROL = 'hsa-miR-3182', 'hsa-miR-184'
C_RED, C_BLUE, C_GOLD, C_GREEN = '#c1443f', '#3a6ea5', '#b8860b', '#2e8b57'
C_ADP, C_UMI, C_ILL = '#b9b9b9', '#a98fc4', '#7d7d7d'


def bea_reads(sample):
    cf = CACHE / f'{sample}.bea.pkl'
    if cf.exists():
        return pickle.loads(cf.read_bytes())
    c = collections.Counter()
    with gzip.open(gg2.ROOT / 'fastq' / 'main' / f'{sample}.fastq.gz', 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 == 1:
                s = line.strip(); p = s.find(ILLUMINA)
                c[s[:p] if p > 0 else s] += 1
    CACHE.mkdir(exist_ok=True); cf.write_bytes(pickle.dumps(c)); return c


def spectrum(counter, ref, L=72):
    keys = [k for k in counter if len(k) >= len(ref)]
    arr = np.full((len(keys), L), 255, np.uint8)
    for i, k in enumerate(keys):
        b = np.frombuffer(k[:L].encode(), np.uint8); arr[i, :len(b)] = b
    w = np.array([counter[k] for k in keys])
    r = np.frombuffer(ref.encode(), np.uint8); k = len(r)
    best = np.zeros(len(keys), np.int16)
    for s in range(L - k + 1):
        np.maximum(best, (arr[:, s:s + k] == r).sum(1).astype(np.int16), out=best)
    return {t: int(w[best == t].sum()) for t in range(k + 1)}, keys, w, best


def trna_lookup(seq):
    """Longest prefix of seq found in a human tRNA; (name, full sequence, prefix length)."""
    f = gg2.ROOT / 'refs' / 'gtrnadb.fasta.gz'
    tr, name = {}, None
    with gzip.open(f, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:] if '9606' in line else None
                if name:
                    tr[name] = ''
            elif name:
                tr[name] += line.upper().replace('U', 'T')
    for L in range(len(seq), 19, -1):
        for k, v in tr.items():
            if seq[:L] in v:
                return k, v, L
    return None, None, 0


def main():
    v22, _ = gg2.refs()
    ref, ctrl_ref = v22[ENTRY], v22[CONTROL]
    counter = bea_reads(SAMPLE)
    bea = gg2.counts('BEA-raw')
    assigned, ctrl_assigned = int(bea.loc[ENTRY, SAMPLE]), int(bea.loc[CONTROL, SAMPLE])
    spec, keys, w, best = spectrum(counter, ref)
    cspec, *_ = spectrum(counter, ctrl_ref)

    mx = best.max()
    idx = np.where(best == mx)[0]; j = idx[np.argmax(w[idx])]
    flag_read = keys[j]
    flag_ins = len(gg2.trim(flag_read) or flag_read)
    molecule = flag_read[:flag_ins]
    tname, tseq, tlen = trna_lookup(molecule)
    short = tname.split('Homo sapiens (human) ')[-1].split(' (')[0] if tname else 'unidentified'
    mism = [i for i in range(len(ref)) if flag_read[i] != ref[i]]
    ctrl_read = max((r for r in counter if r.startswith(ctrl_ref)), key=lambda r: counter[r])
    print(f'{ENTRY}: assigned {assigned}, exact {spec[len(ref)]}, '
          f'near {spec[len(ref)-1]:,}; molecule {flag_ins} nt = {short} ({tlen} nt matched)')

    RL = max(len(ctrl_read), len(flag_read))
    fig = plt.figure(figsize=(15, 11.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.92, 1.80, 0.80], hspace=.34,
                          left=.065, right=.965, top=.862, bottom=.055)
    fig.suptitle(f'Artifact example: {ENTRY}', fontsize=17, weight='bold', y=.962)
    fig.text(.5, .917, f'sample {SAMPLE}  ·  BEA-raw: facility counts, STAR + featureCounts, miRBase v22',
             ha='center', va='center', fontsize=10.5, color='#555')

    # ---- Row 1: match spectra ----
    sub = gs[0].subgridspec(1, 2, wspace=.20)
    for col, (name, sp, asg, rf) in enumerate([(ENTRY, spec, assigned, ref),
                                               (CONTROL, cspec, ctrl_assigned, ctrl_ref)]):
        ax = fig.add_subplot(sub[0, col])
        # a 17-mer matches 13-14 bases by chance constantly, so the window is tighter than in the miR-5701 case study
        k = len(rf); ts = list(range(k - 2, k + 1))
        vals = [sp.get(t, 0) for t in ts]
        base = C_RED if col == 0 else C_BLUE
        cols = [base] * len(ts); cols[-1] = C_GREEN if vals[-1] > 10 else '#dcdcdc'
        ax.bar([str(t) for t in ts], vals, color=cols, edgecolor='#666', linewidth=.6)
        ax.axhline(asg, color='k', ls='--', lw=1.3)
        ax.text(.01, asg, f' BEA counted {asg:,}', fontsize=9.5, va='bottom',
                transform=ax.get_yaxis_transform())
        ax.set_xlabel(f'bases matching the {k} nt reference', fontsize=9.5)
        ax.set_ylabel('reads', fontsize=9.5)
        ax.set_title(f'{name} — {vals[-1]:,} exact matches',
                     fontsize=11.5, weight='bold', color=base, pad=8)
        ax.grid(axis='y', alpha=.3); ax.set_axisbelow(True)
        ax.set_ylim(0, max(max(vals), asg) * 1.14)

    # ---- Row 2: 72-cycle reads ----
    ax = fig.add_subplot(gs[1]); ax.axis('off')
    ax.set_xlim(-15, RL + 15); ax.set_ylim(0, 1)
    ax.text(-15, 1.03, '72-cycle reads', fontsize=13, weight='bold', va='top')

    def draw(y, rd, ins_len, rna_col, tag):
        segs = [(0, ins_len, rna_col)]
        a = min(len(rd), ins_len + len(gg2.ADAPTER)); segs.append((ins_len, a, C_ADP))
        if a < len(rd):
            u = min(len(rd), a + UMI_LEN); segs.append((a, u, C_UMI))
            if u < len(rd):
                segs.append((u, len(rd), C_ILL))
        names = {rna_col: 'the RNA molecule', C_ADP: 'QIAseq adapter',
                 C_UMI: 'UMI', C_ILL: 'Illumina adapter'}
        for s0, s1, c in segs:
            ax.add_patch(Rectangle((s0 - .5, y - .038), s1 - s0, .076, facecolor=c,
                                   edgecolor='#666', lw=.7, zorder=1))
            if s1 - s0 >= 9:
                ax.text((s0 + s1 - 1) / 2, y - .062, names[c], fontsize=8.4,
                        ha='center', va='top', color='#444')
        for i, b in enumerate(rd):
            ax.text(i, y, b, family='monospace', fontsize=7.2, ha='center', va='center',
                    color='w' if i < ins_len else '#333', zorder=3)
        ax.text(-1.8, y, tag, fontsize=10.5, ha='right', va='center', weight='bold')
        ax.annotate('', xy=(-.5, y + .058), xytext=(ins_len - .5, y + .058),
                    arrowprops=dict(arrowstyle='<->', color=rna_col, lw=1.6))
        ax.text((ins_len - 1) / 2, y + .076, f'{ins_len} nt', fontsize=10.5,
                ha='center', color=rna_col, weight='bold')

    draw(.84, ctrl_read, len(ctrl_ref), C_BLUE, CONTROL)
    draw(.54, flag_read, flag_ins, C_GOLD, f'{ENTRY} suspect')
    ax.plot([RL - .5, RL - .5], [.48, .90], color=C_RED, lw=2.2)
    ax.text(RL + .8, .69, 'cycle 72\nread stops', fontsize=9.5, color=C_RED,
            weight='bold', va='center')

    def refrow(y, seq, start, tag, note, col, trail=''):
        for i, b in enumerate(seq):
            x = start + i
            inside = 0 <= x < len(flag_read)
            bad = inside and flag_read[x] != b
            ax.text(x, y + .048, ('✗' if bad else '|') if inside else '',
                    family='monospace', fontsize=8.4, ha='center', va='center',
                    color=C_RED if bad else '#aaa')
            ax.text(x, y, b, family='monospace', fontsize=7.2, ha='center', va='center',
                    color=C_RED if bad else ('#bbb' if not inside else '#333'),
                    weight='bold' if bad else 'normal')
        ax.text(start - 1.8, y, tag, fontsize=9.5, ha='right', va='center', weight='bold')
        ax.text(start + len(seq) + .8, y, note, fontsize=9.5, va='center', color=col,
                weight='bold')
        if trail:
            ax.text(start + len(seq) + .8, y - .052, trail, fontsize=8.5,
                    va='center', color='#888')

    shown = tseq[:flag_ins] if tseq else ''
    refrow(.30, shown, 0, f'{short}', 'matches end to end', C_GREEN,
           trail=f'(tRNA is {len(tseq)} nt; the rest is past cycle 72)' if tseq else '')
    refrow(.12, ref, 0, f'{ENTRY} reference',
           f'{len(ref)-len(mism)}/{len(ref)}', C_RED)
    ax.text(len(ref) / 2 - .5, .045,
            f'differs at the final base only — {spec[len(ref)-1]:,} reads match {len(ref)-1} of {len(ref)}',
            fontsize=9.5, ha='center', color=C_RED, weight='bold')

    # ---- Row 3: per branch ----
    ax = fig.add_subplot(gs[2])
    M = gg2.name_map_v22_v23()
    arms, vals = [], []
    for a in gg2.ARMS:
        m = gg2.counts(a); ver = gg2.ARMS[a][1]
        k = ENTRY if ENTRY in m.index else (M.get(ENTRY, (None,))[0] if ver == 'v23' else None)
        arms.append(a)
        vals.append(float((m.loc[k] / m.sum() * 1e6).mean()) if k and k in m.index else 0.0)
    cols = [C_RED if a.startswith('BEA') else C_GREEN for a in arms]
    ax.barh(range(len(arms)), vals, color=cols, edgecolor='#666', linewidth=.6)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(f'mean CPM assigned to {ENTRY} (all 95 samples)', fontsize=9.5)
    ax.set_title('alignment counts by method', fontsize=11.5, weight='bold')
    ax.set_xlim(0, max(vals) * 1.16)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .012, i, f'{v:,.0f}' if v else '0', va='center', fontsize=9.5)
    ax.grid(axis='x', alpha=.3); ax.set_axisbelow(True)

    fig.savefig(OUT / 'fig_case_mir3182.png', dpi=150)
    print(f'wrote {OUT}/fig_case_mir3182.png')


if __name__ == '__main__':
    main()
