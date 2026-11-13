#!/usr/bin/env python3
"""Case study: how hsa-miR-5701 gets 655 reads it should not have; reads reconstructed the way BEA had them (Illumina adapter stripped, QIAseq adapter + UMI kept)."""
import sys, gzip, pickle, collections, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import gg2

# gg2.counts() not gg2.matrix(): the case study is ABOUT an excluded entry

def snord_lookup(seq):
    """Name the molecule from the vendored snoRNA databases (RNAcentral snoDB/snoPY)."""
    import gzip
    for db in ['snopy', 'snodb']:
        f = gg2.ROOT / 'refs' / f'{db}.fasta.gz'
        if not f.exists():
            continue
        name, cur = None, ''
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('>'):
                    if name and seq in cur:
                        return name, cur
                    name, cur = line[1:], ''
                else:
                    cur += line.upper().replace('U', 'T')
        if name and seq in cur:
            return name, cur
    return None, None


OUT = gg2.PLOTS / 'stage_1'; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / '.insert_cache'; CACHE.mkdir(exist_ok=True)
SAMPLE = 'BEA25P077_RP_100_40A_G2'
ILLUMINA = 'AGATCGGAAGAGC'
ENTRY, CONTROL = 'hsa-miR-5701', 'hsa-miR-184'
UMI_LEN, ILLUMINA = 12, 'AGATCGGAAGAGC'

C_RED, C_BLUE, C_GREY, C_GOLD, C_GREEN = '#c1443f', '#3a6ea5', '#9a9a9a', '#b8860b', '#2e8b57'


def bea_reads(sample):
    """Reads as BEA had them: Illumina adapter stripped, QIAseq adapter + UMI kept."""
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


def raw_examples(sample, ctrl_seq, core):
    """Full 72 nt reads (nothing stripped) for the anatomy diagram."""
    ctrl = flag = None
    with gzip.open(gg2.ROOT / 'fastq' / 'main' / f'{sample}.fastq.gz', 'rt') as fh:
        for i, line in enumerate(fh):
            if i % 4 != 1:
                continue
            r = line.strip()
            if ctrl is None and r.startswith(ctrl_seq):
                ctrl = r
            if flag is None and core in r:
                flag = r
            if ctrl and flag:
                break
    return ctrl, flag


def spectrum(counter, ref, L=72):
    """Reads binned by best ungapped local-alignment score against ref."""
    keys = [k for k in counter if len(k) >= len(ref)]
    arr = np.full((len(keys), L), 255, np.uint8)
    for i, k in enumerate(keys):
        b = np.frombuffer(k[:L].encode(), np.uint8); arr[i, :len(b)] = b
    w = np.array([counter[k] for k in keys])
    r = np.frombuffer(ref.encode(), np.uint8); k = len(r)
    best = np.zeros(len(keys), np.int16)
    for s in range(L - k + 1):
        np.maximum(best, (arr[:, s:s + k] == r).sum(1).astype(np.int16), out=best)
    return {t: int(w[best == t].sum()) for t in range(k + 1)}, w.sum()


def top_read(counter, ref):
    """Most abundant read aligning to ref with <=1 mismatch, and its window."""
    best = None
    for read, n in counter.items():
        if len(read) < len(ref):
            continue
        for s in range(len(read) - len(ref) + 1):
            win = read[s:s + len(ref)]
            mm = [i for i in range(len(ref)) if win[i] != ref[i]]
            if len(mm) <= 1 and (best is None or n > best[0]):
                best = (n, read, s, mm)
            break_ = len(mm) <= 1
            if break_:
                break
    return best


def main():
    v22, _ = gg2.refs()
    ref, ctrl_ref = v22[ENTRY], v22[CONTROL]
    counter = bea_reads(SAMPLE)
    bea = gg2.counts('BEA-raw')
    assigned = int(bea.loc[ENTRY, SAMPLE])
    ctrl_assigned = int(bea.loc[CONTROL, SAMPLE])

    spec, _ = spectrum(counter, ref)
    cspec, _ = spectrum(counter, ctrl_ref)
    n, read, off, mm = top_read(counter, ref)
    fam = sum(v for k, v in counter.items() if read[20:50] in k)
    # reads often end mid-adapter, so fall back to the longest adapter prefix at the 3' end
    p = read.find(gg2.ADAPTER)
    if p < 0:
        for L in range(len(gg2.ADAPTER), 7, -1):
            q = read.find(gg2.ADAPTER[:L])
            if q > 0 and q + L == len(read):
                p = q; break
    molecule = read[:p] if p > 0 else read
    win_start = off

    print(f'{ENTRY}: BEA assigned {assigned}; exact matches {spec[len(ref)]}; '
          f'1-mismatch {spec[len(ref)-1]}; molecule {len(molecule)} nt; family {fam} reads')

    ctrl_seq = v22[CONTROL]
    sname, sfull = snord_lookup(molecule)
    label = 'SNORD43' if sname and 'SNORD43' in sname else (sname or 'unidentified')
    ctrl_read, flag_read = raw_examples(SAMPLE, ctrl_seq, molecule[20:50])
    ctrl_ins = len(ctrl_seq)
    flag_ins = flag_read.find(gg2.ADAPTER)
    if flag_ins < 0:
        for L in range(len(gg2.ADAPTER) - 1, 5, -1):
            q = flag_read.find(gg2.ADAPTER[:L])
            if q > 0 and q + L == len(flag_read):
                flag_ins = q; break
    win = flag_read.find(flag_read[flag_ins - len(ref):flag_ins])
    RL = max(len(ctrl_read), len(flag_read))
    C_ADP, C_UMI, C_ILL = '#b9b9b9', '#a98fc4', '#7d7d7d'
    print(f'  ctrl read {len(ctrl_read)} nt (insert {ctrl_ins}); '
          f'flag read {len(flag_read)} nt (insert {flag_ins})')

    fig = plt.figure(figsize=(15, 11.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.92, 1.80, 0.80], hspace=.34,
                          left=.065, right=.965, top=.862, bottom=.055)
    fig.suptitle('Artifact example: hsa-miR-5701', fontsize=17, weight='bold', y=.962)
    fig.text(.5, .917, f'sample {SAMPLE}  ·  BEA-raw: facility counts, STAR + featureCounts, miRBase v22',
             ha='center', va='center', fontsize=10.5, color='#555')

    # Row 1: match spectra
    sub = gs[0].subgridspec(1, 2, wspace=.20)
    for col, (name, sp, asg, rf) in enumerate([(ENTRY, spec, assigned, ref),
                                               (CONTROL, cspec, ctrl_assigned, ctrl_ref)]):
        ax = fig.add_subplot(sub[0, col])
        k = len(rf); ts = list(range(k - 4, k + 1))
        vals = [sp.get(t, 0) for t in ts]
        base = C_RED if col == 0 else C_BLUE
        cols = [base] * len(ts); cols[-1] = C_GREEN if vals[-1] > 0 else '#dcdcdc'
        ax.bar([str(t) for t in ts], vals, color=cols, edgecolor='#666', linewidth=.6)
        ax.axhline(asg, color='k', ls='--', lw=1.3)
        ax.text(0.01, asg, f' BEA counted {asg:,}', fontsize=9.5, va='bottom',
                transform=ax.get_yaxis_transform())
        ax.set_xlabel(f'bases matching the {k} nt reference', fontsize=9.5)
        ax.set_ylabel('reads', fontsize=9.5)
        exact = vals[-1]
        ax.set_title(f'{name} — {exact:,} exact matches',
                     fontsize=11.5, weight='bold', color=base, pad=8)
        ax.grid(axis='y', alpha=.3); ax.set_axisbelow(True)
        ax.set_ylim(0, max(max(vals), asg) * 1.14)

    # Row 2: 72-cycle reads
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
                ax.text((s0 + s1 - 1) / 2, y - .062, names[c], fontsize=8.4, ha='center',
                        va='top', color='#444')
        for i, b in enumerate(rd):
            ax.text(i, y, b, family='monospace', fontsize=7.2, ha='center', va='center',
                    color='w' if i < ins_len else '#333', zorder=3)
        ax.text(-1.8, y, tag, fontsize=10.5, ha='right', va='center', weight='bold')
        ax.annotate('', xy=(-.5, y + .058), xytext=(ins_len - .5, y + .058),
                    arrowprops=dict(arrowstyle='<->', color=rna_col, lw=1.6))
        ax.text((ins_len - 1) / 2, y + .076, f'{ins_len} nt', fontsize=10.5, ha='center',
                color=rna_col, weight='bold')

    draw(.84, ctrl_read, ctrl_ins, C_BLUE, 'hsa-miR-184')
    draw(.54, flag_read, flag_ins, C_GOLD, 'hsa-miR-5701 suspect')
    ax.plot([RL - .5, RL - .5], [.48, .90], color=C_RED, lw=2.2)
    ax.text(RL + .8, .69, 'cycle 72\nread stops', fontsize=9.5, color=C_RED,
            weight='bold', va='center')

    def refrow(y, seq, start, tag, note, col):
        for i, b in enumerate(seq):
            x = start + i
            inside = 0 <= x < len(flag_read)
            bad = inside and flag_read[x] != b
            mark = ('✗' if bad else '|') if inside else ''
            ax.text(x, y + .048, mark, family='monospace', fontsize=8.4, ha='center',
                    va='center', color=C_RED if bad else '#aaa')
            ax.text(x, y, b, family='monospace', fontsize=7.2, ha='center', va='center',
                    color=C_RED if bad else ('#bbb' if not inside else '#333'),
                    weight='bold' if bad else 'normal')
        ax.text(start - 1.8, y, tag, fontsize=9.5, ha='right', va='center', weight='bold')
        ax.text(start + len(seq) + .8, y, note, fontsize=9.5, va='center', color=col,
                weight='bold')

    snord_start = -sfull.find(molecule)          # SNORD43 begins before the read's 5' end
    refrow(.30, sfull, snord_start, f'{label} reference', 'matches end to end', C_GREEN)
    for lab, pat, only3p in [('C box', '[AG]TGATGA', False), ('D box', 'CTGA', True)]:
        for m in re.finditer(pat, sfull):
            if only3p and m.start() < len(sfull) - 8:
                continue
            a, b_ = snord_start + m.start(), snord_start + m.end()
            ax.add_patch(Rectangle((a - .5, .30 - .030), b_ - a, .060, facecolor='none',
                                   edgecolor='#5c3d00', lw=1.9, zorder=4))
            ax.text((a + b_ - 1) / 2, .30 - .048, lab, fontsize=8.6, ha='center',
                    va='top', color='#5c3d00', weight='bold', zorder=5)
    ax.text(snord_start + len(sfull) + .8, .30 - .050,
            'C box + D box = the motifs\nthat identify a C/D box snoRNA',
            fontsize=8.6, va='top', color='#5c3d00')

    refrow(.12, ref, win, 'miR-5701 reference', f'{len(ref)-len(mm)}/{len(ref)}', C_RED)
    ax.text(win + len(ref) / 2 - .5, .045,
            f'651 of the 652 reads carry the same mismatch, at position {mm[0]+1}',
            fontsize=9.5, ha='center', color=C_RED, weight='bold')

    # Row 3: which pipelines report it
    ax = fig.add_subplot(gs[2])
    arms, vals = [], []
    for a in gg2.ARMS:
        m = gg2.counts(a)
        hit = ENTRY if ENTRY in m.index else next(
            (c for c in m.index if c.startswith(ENTRY + '-')), None)
        arms.append(a); vals.append(float((m.loc[hit] / m.sum() * 1e6).mean()) if hit else 0.0)
    cols = [C_RED if a.startswith('BEA') else C_GREEN for a in arms]
    ax.barh(range(len(arms)), vals, color=cols, edgecolor='#666', linewidth=.6)
    ax.set_yticks(range(len(arms))); ax.set_yticklabels(arms, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel('mean CPM assigned to miR-5701 (all 95 samples)', fontsize=9.5)
    ax.set_title('alignment counts by method', fontsize=11.5, weight='bold')
    ax.set_xlim(0, max(vals) * 1.16)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * .012, i, f'{v:,.1f}' if v else '0', va='center', fontsize=9.5)
    ax.grid(axis='x', alpha=.3); ax.set_axisbelow(True)

    fig.savefig(OUT / 'fig_case_mir5701.png', dpi=150)
    print(f'wrote {OUT}/fig_case_mir5701.png')


if __name__ == '__main__':
    main()
