#!/usr/bin/env python3
"""Build manuscript/draft/02_methods_results.ipynb - the Methods and Results draft."""
import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'manuscript' / 'draft' / '02_methods_results.ipynb'

CELLS = []


def md(text):
    CELLS.append(('md', text.strip('\n')))


def code(text):
    CELLS.append(('code', text.strip('\n')))


# front matter

md(r'''
# GlauGene2 — Methods and Results

**Draft, generated 2026-10-07 from `scripts/manuscript/build_notebook.py`.**
miRNA sequencing of exfoliative glaucoma (XFG) versus cataract-only controls in
anterior lens capsule and tear fluid.

---

### How to compile this into manuscript text

Every analysis appears as a **markdown cell followed by one code cell**:

* the markdown cell is the prose — method description, or the framing of a result;
* the code cell prints the numbers, as complete sentences, read from the analysis
  output tables.

To produce the sections, join the two: take the markdown, substitute the printed
sentences where the prose refers to a quantity, and keep the order. **Do not
recompute, reinterpret or round anything**: the printed values are the ones the
figures were drawn from.

Three conventions used throughout:

* `METHODS →` and `RESULTS →` prefixes in printed output mark which section a
  sentence belongs to when a cell serves both.
* `CAVEAT →` marks something that must survive into the text. These are not
  optional hedges; each one is a design fact a reviewer will otherwise find.
* `[APPENDIX]` in a heading means supplementary, not main text.

Figures are referenced by relative path and will render in Jupyter. Main-text
figures are `manuscript/figures/figure1..6`; appendix figures are referenced
where they belong.

### Scope

Main text carries Figures 1–6. The appendix carries the covariate PCAs, the
clustered heatmap, permutation calibration, equivalence and power, stability,
small-study resampling, the classifier variants, the replication arms, the
enrichment work, and the per-miRNA tables. All of it is below.
''')

code(r'''
import os, sys, textwrap, warnings
from pathlib import Path

warnings.filterwarnings('ignore')
ROOT = Path.cwd()
while not (ROOT / 'scripts' / 'common' / 'gg2.py').exists():
    ROOT = ROOT.parent                      # run from anywhere inside the repo
os.environ.setdefault('GG2_ARM', 'BEA-dedup')
for sub in ('common', 'stage_2', 'stage_3', 'manuscript'):
    sys.path.insert(0, str(ROOT / 'scripts' / sub))

import gg2, pca
from de import filtered as de_filtered
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ARM = gg2.PRIMARY
P = gg2.PLOTS                 # plots/branches/BEA-dedup
SHARED = gg2.SHARED           # plots/  (pipeline-independent outputs)
REFS = ROOT / 'refs'

MD = gg2.metadata()
MX = gg2.matrix(ARM)
MD = MD.loc[list(MX.columns)]                # analysed samples only
TISSUE = {t: MD.index[MD.tissue_name == t] for t in ('lens', 'tear')}
# the study's single entry filter (M7), used by the tests, the PCA and the heatmap alike
KEEP = {t: set(de_filtered(MX, MD, t)[0].index) for t in ('lens', 'tear')}
KEEP['joint'] = KEEP['lens'] | KEEP['tear']


def tsv(stage, name, branch=None):
    base = ROOT / 'plots' / 'branches' / branch if branch else P
    return pd.read_csv(base / stage / name, sep='\t')


def say(s):
    print(textwrap.fill(' '.join(s.split()), 94,
                        subsequent_indent='    '))


def head(t):
    print(t)
    print('=' * len(t))


print(f'arm = {ARM}   samples = {MX.shape[1]}   entries in matrix = {MX.shape[0]}')
''')

# methods cells

md('''
---
# Part 1 — Methods
---
''')

md('''
## M1. Study design and participants

Patients undergoing routine cataract surgery at a single centre were enrolled in two
groups: exfoliative glaucoma (XFG) and cataract-only controls without exfoliation or
glaucoma. Every case carried a clinical diagnosis of exfoliative **glaucoma**, not
exfoliation syndrome alone. Two specimens were collected per patient where possible —
the anterior lens capsule excised during capsulorhexis, and tear fluid — so that most
patients contribute to both tissue analyses and the lens-versus-tear contrast can be
made within patient.

Clinical covariates recorded for each patient were sex, age, Schirmer score,
exfoliation status, diabetes, hypertension, dementia, vitamin supplementation,
smoking status and cigarettes per day, family history of glaucoma, Swedish origin,
and IOP-lowering eyedrop use.

**The disease contrast is perfectly collinear with eyedrop exposure.** Every XFG
patient was on IOP-lowering therapy and no control was. No analysis in this paper can
separate exfoliative glaucoma from its treatment, and every disease claim should be
read as "XFG or its treatment". The cohort is also ancestry-homogeneous, which removes
ancestry as a confounder but means the result does not transfer to other populations
without checking.
''')

code(r'''
head('M1 / R0  cohort')
n_pat = MD.patient.nunique()
per_pat = MD.groupby('patient')['tissue_name'].nunique()
both = per_pat[per_pat == 2].index
one_pat = MD[MD.patient.isin(both)].drop_duplicates('patient').disease.value_counts()

say(f'METHODS -> Lens capsule and tear fluid were profiled from {n_pat} patients. '
    f'After the exclusions described in M5 and M6, {len(TISSUE["lens"])} lens and '
    f'{len(TISSUE["tear"])} tear libraries entered analysis.')
for t in ('lens', 'tear'):
    s = MD.loc[TISSUE[t]]
    say(f'METHODS -> {t.capitalize()}: n = {len(s)} '
        f'({int((s.disease == "diseased").sum())} XFG, '
        f'{int((s.disease == "control").sum())} control).')
say(f'METHODS -> {len(both)} patients gave both tissues '
    f'({int(one_pat.get("control", 0))} control, {int(one_pat.get("diseased", 0))} XFG); '
    f'these are the patients used for the paired tissue contrast (M18).')

tab = pd.crosstab(MD.disease, MD.eyedrops.notna().map({True: 'on drops', False: 'no drops'}))
print()
print(tab.to_string())
say('CAVEAT -> Eyedrop exposure is perfectly collinear with disease, as the table '
    'shows. State this once, prominently, in the Discussion and carry it into every '
    'permutation caveat.')
''')

md('''
## M2. Specimen collection

Anterior lens capsule was obtained as the capsulorhexis button during phacoemulsification
and transferred directly to preservative. Tear fluid was collected before surgery.
Schirmer strips were used for tear volume measurement; the Schirmer value is retained
as a covariate because tear collection volume plausibly affects library composition.
No additional procedure was required of any patient for either specimen: the capsule is
discarded tissue from the operation, and tear sampling is non-invasive.

*(Collection detail — anaesthetic, preservative, strip handling, storage temperature and
time to extraction — must be filled in from the clinical protocol. It is not recoverable
from the sequencing data and is not in this repository.)*
''')

md('''
## M3. Library preparation and sequencing

Small RNA libraries were prepared with the QIAseq miRNA Library Kit, which ligates a
3′ adapter (`AACTGTAGGCACCATCAAT`) followed by a 12 nt unique molecular identifier, and
sequenced on an Illumina NovaSeq. Reads are single-end 72 nt, so the insert length is
recoverable from the position of the adapter within the read rather than from the read
length.

*(Kit version, input amount, cycle number, PCR cycles, pooling and lane layout to be
taken from the facility report.)*
''')

code(r'''
head('M3 / R0  sequencing depth and library composition')
rc = pd.read_csv(SHARED / 'stage_1' / 'rna_composition.tsv', sep='\t').set_index('sample')
rc = rc.loc[[s for s in MX.columns if s in rc.index]]
for t in ('lens', 'tear'):
    s = rc.loc[[i for i in TISSUE[t] if i in rc.index]]
    say(f'RESULTS -> {t.capitalize()}: median {s.total_reads.median()/1e6:.1f} M reads '
        f'(range {s.total_reads.min()/1e6:.1f}-{s.total_reads.max()/1e6:.1f} M); '
        f'miRNA made up a median {100*(s.miRNA/s.classified).median():.1f}% of '
        f'classified reads '
        f'({100*(s.miRNA/s.classified).min():.1f}-{100*(s.miRNA/s.classified).max():.1f}%).')
say('RESULTS -> The two tissues differ markedly in library purity: tear libraries carry '
    'a large tRNA-fragment and rRNA component, so a far smaller share of a tear library '
    'is miRNA than of a lens library. This is a property of the specimen, not of the '
    'pipeline, and it is why the two tissues are normalised and tested separately '
    'throughout.')
print()
print((rc.assign(tissue=MD.loc[rc.index, 'tissue_name'])
         .groupby('tissue')[['miRNA', 'tRNA', 'rRNA', 'snoRNA', 'piRNA', 'YRNA']]
         .apply(lambda g: (g.div(rc.loc[g.index, 'classified'], axis=0) * 100).median())
         .round(1).to_string())
      + '\n   (median % of classified reads)')
''')

md('''
## M4. Read quantification

Four published or facility pipelines were run over the same FASTQ files, plus the
facility's own delivered counts, giving the arms below. Running several was not a
robustness afterthought: two of them exist to reproduce a comparator study's
quantification exactly, and the remainder establish that the conclusion does not
depend on the pipeline.

| arm | method | miRBase | role |
|---|---|---|---|
| `BEA-dedup` | facility counts, STAR + featureCounts, UMI-deduplicated | v22 | **primary** |
| `BEA-raw` | the same counts without deduplication | v22 | sensitivity |
| `nfcore` | nf-core/smrnaseq, fastp → bowtie1 `-n 2` → mirtop (isomiR-aware) | v23 | sensitivity |
| `nfcore-dedup` | as above, deduplicated on (insert, UMI) by UMICollapse | v23 | sensitivity |
| `CBGM-v22` | Cutadapt 4.0 → bowtie1 `-v 0`, 21–24 nt, mature only | v22 | replicates the tear miRNome reference |
| `PEXlens-v22` | Cutadapt 4.0 → bowtie2 defaults, mature only | v22 | replicates the lens-capsule comparator |

`BEA-dedup` is the pre-specified primary arm. Its deduplication rule is the facility's
rather than ours — directional, edit distance 2, per position — and it was characterised
before being accepted. The two replication arms were rebuilt against miRBase v22 because
both comparator studies used v22; mapping names between releases by string would
silently fail (`hsa-miR-184` against `hsa-miR-184-3p`), so every cross-release comparison
maps by mature sequence first.

**Primary-arm history must be declared, not smoothed over.** `BEA-raw` was primary at an
earlier stage and the switch to `BEA-dedup` was made after results existed. The
cross-arm comparison table is reported in full for exactly this reason.
''')

code(r'''
head('M4 / R-appendix  the conclusion across quantification arms')
ac = pd.read_csv(SHARED / 'arm_comparison.tsv', sep='\t')
print(ac.to_string(index=False))
print()
say('RESULTS -> Across every arm and both tissues, the number of miRNAs surviving '
    'Benjamini-Hochberg correction is 0 or 1, the permutation p-value for the observed '
    'hit count is never small, and the classifier AUC never separates from its null. '
    'The one arm-dependent item is miR-708-5p in tear, which clears BH correction under '
    'the primary arm and not under all others; it is reported as such.')
say('CAVEAT -> UMI deduplication is the one pipeline choice that is genuinely '
    'unsettled. nf-core/smrnaseq does not deduplicate on the QIAseq UMI by default, so '
    'the nfcore arm counts PCR duplicates as independent reads; nfcore-dedup and '
    'BEA-dedup do deduplicate, by different rules. See manuscript/notes/'
    'umi_deduplication.md for the comparison.')
''')

md('''
## M5. Read-level verification of miRNA assignments

Short-read aligners will assign reads to a miRBase entry whose sequence is not present
in the library, if a longer RNA contains a near-match. Entries flagged as suspicious by
the facility, and entries whose counts differed by more than five-fold between pipelines,
were therefore checked directly against the reads: for each entry, how many of the
pooled 49.9 M reads actually contain its mature sequence?

Six abundant, uncontested miRNAs were used as positive controls and sit at an
observed-to-assigned ratio of 0.45–0.95. Twenty-eight entries sit at approximately zero
and were excluded from all analyses:

* **18 entries whose sequence is absent**: the assigned counts are alignment artefacts.
  `miR-5701` resolves to SNORD43 and `miR-4791` to a human snoRNA.
* **9 entries inflated rather than invented**: the entry is reported by the strict
  pipelines too, but the facility arm assigns it 5–380× more, and the sequence has zero
  standalone molecules. `miR-3182` is the largest: 17 nt, the 5′ half of tRNA-Val(CAC)
  with one substitution, carrying 4.9% of the facility library on 2 exact matches
  against 199,504 reads matching 16 of its 17 bases.
* **1 annotation error**: `miR-1246`, present only inside a longer transcript.

`miR-4485-5p` is 16 nt, short enough to occur inside longer molecules by chance, so the
read-level test cannot resolve it; it is retained and flagged rather than excluded.

Exclusion is applied to our own analyses throughout. It is **not** applied to the
replication arms, where the comparator pipelines are reproduced as published and had no
such list — that difference is itself a methodological finding, since those pipelines
quantify tRNA and snoRNA fragments as miRNA.
''')

code(r'''
head('M5 / R-appendix  verified mis-assignments')
sv = pd.read_csv(SHARED / 'stage_1' / 'suspects_verified.tsv', sep='\t')
print(sv.groupby('verdict').agg(n=('entry', 'size'),
                                median_ratio=('ratio', 'median')).round(4).to_string())
ctrl = sv[sv.verdict == 'control']
say(f'METHODS -> Positive controls (n = {len(ctrl)}) had observed/assigned ratios of '
    f'{ctrl.ratio.min():.2f}-{ctrl.ratio.max():.2f}.')
say(f'METHODS -> The verified table above covers the {len(sv)} entries put through the '
    f'read-level test directly. The nine inflated entries were identified by a separate '
    f'cross-pipeline ratio scan and verified the same way, which is why the exclusion '
    f'list is longer than that table.')
say(f'METHODS -> {len(gg2.EXCLUDE)} entries were excluded: '
    f'{len(gg2.ARTIFACTS_ABSENT)} absent, {len(gg2.ARTIFACTS_INFLATED)} inflated, '
    f'{len(gg2.ARTIFACTS_ANNOTATION)} annotation error. '
    f'{len(gg2.UNRESOLVED)} entry ({", ".join(gg2.UNRESOLVED)}) was unresolvable at '
    f'16 nt and retained with a flag.')
m_raw = gg2.counts(ARM)
frac = m_raw.loc[[e for e in gg2.EXCLUDE if e in m_raw.index]].sum().sum() / m_raw.sum().sum()
say(f'RESULTS -> Together the excluded entries carry {100*frac:.2f}% of the '
    f'{ARM} library.')
say('CAVEAT -> A further set of entries is a blind spot of the facility arm rather than '
    'an artefact: every miRBase v22 entry with a reverse-complement twin is reported as '
    'zero in all samples, because a read matches both features and featureCounts '
    'discards ambiguous assignments. Reporting those as zero expression would be a false '
    f'negative, so the {len(gg2.BLIND_SPOTS[ARM])} affected entries are dropped for this '
    'arm and kept elsewhere.')
''')

md('''
## M6. Sample-level exclusion

One tear library was excluded. The criterion was **agreement with its peers**, not read
quality: for each sample, the Spearman correlation of its miRNA profile with the median
profile of its own tissue. Sample `40A` sits at 0.730 against a tear median of 0.936,
and is the only library in the cohort below 0.85 — the next worst is 0.875. It also
detects the fewest miRNAs of any sample (196 against a median of 259), its library has
collapsed onto a few species (top-5 concentration 59% against 30%), and it is a
singleton in the unsupervised clustering.

Peer agreement was chosen deliberately over a read-quality rule. `40A` does have a high
adapter-less fraction (15.8%), but four lens libraries are as bad or worse on that
measure while agreeing with their peers normally, so a read-quality threshold would
discard usable data. All 95 libraries are retained in the QC figures that document the
exclusion.
''')

md('''
## M7. Expression filtering and normalisation

Within each tissue, an entry was tested if it reached **≥10 reads in ≥50% of either
disease group** in that tissue (`scripts/stage_3/de.py:filtered`). The rule is a
disjunction, not a conjunction: an entry expressed in half the controls and in none of
the cases is kept, because that is exactly the pattern a disease effect would produce and
a both-groups rule would discard it. This gives 226 testable entries in lens and 259 in
tear.

The filter is applied once, per tissue, on the primary arm, and held fixed across every
downstream analysis including the permutations — so that every run tests the same feature
set and hit counts are comparable between the real labels and the shuffled ones.

**This one filter now governs the unsupervised analyses too** (M17). Until 2026-10-12 the
PCA used a looser, label-blind rule of its own and reported 258 and 302 entries; it was
standardised onto this filter after checking that nothing rests on the choice.

Two families of normalisation were carried through in parallel, because they ask
different questions:

* **relative**: CPM, TMM and median-of-ratios. These test composition: whether a miRNA
  makes up a different share of the library between groups.
* **absolute**: size factors from total FASTQ reads, which preserves any global shift in
  how much miRNA a specimen yields rather than normalising it away.

CPM under the relative family is the reported default. Size factors are supplied
explicitly to every testing method, so all of them run through identical dispersion
estimation and differ only in the model.
''')

md('''
## M8. Differential expression

Lens and tear were analysed separately, with the design

    ~ sex + disease

on filtered counts. Disease is the coefficient of interest. Patient could not be blocked
in this model because each patient contributes at most one sample per tissue; the paired
structure is used in the tissue contrast (M18) instead.

Four testing methods were run over the same counts, filter, design and size factors:

* **edgeR quasi-likelihood F-test**: the lead method, and edgeR's current
  recommendation. Reported throughout unless stated otherwise.
* **edgeR likelihood-ratio test**: included because it is what the lens-capsule
  comparator used.
* **negative-binomial GLM with Wald test** (pydeseq2).
* **NBSR**, a Dirichlet-multinomial model that treats the library as compositional.

Reporting one lead method with three comparators, rather than picking the most favourable,
is the point: a negative result that holds under four models is harder to attribute to
model choice. The agreement is reported in full in the appendix.
''')

md('''
### M8b. Which engine runs which analysis. Check this before writing "the primary model"

Not every analysis in this paper runs on edgeR-QL, and the difference has already caused
two wrong sentences in the drafts. The table is the authority; the prose must match it.
**Normalisation is the second axis on which errors happen** and is now a column: the paired
tissue contrast ran on TMM until 2026-10-19 while everything else ran on CPM, which went
unnoticed in two manuscript passes. Every reported analysis now uses CPM except the lens
replication arm, which uses TMM because the comparator did.

This table covers what the notebook computes. The manuscript reports a subset: pydeseq2 and
the Box-Cox model are dropped from it entirely, as is the simulated power grid.

| analysis | engine | normalisation | design | section |
|---|---|---|---|---|
| differential expression (reported) | **edgeR-QL** | **CPM** | `~ sex + disease` | M8 / R3 |
| model sensitivity grid | edgeR-LRT, limma-voom, NBSR, pydeseq2, Box-Cox | CPM, TMM, median-of-ratios, input-RNA | `~ sex + disease` | R3b |
| permutation null | **edgeR-QL** (1,000 runs); pydeseq2 Wald (60-run arm) | **CPM** | `~ sex + disease` | M10 / R5 |
| equivalence / TOST | **edgeR-QL** (lead); pydeseq2, NBSR as comparators | **CPM** | `~ sex + disease` | M11 / R6 |
| stability (LOO, random-3) | **edgeR-QL** | **CPM** | `~ sex + disease` | M13 / R7 |
| power, simulated grid | **pydeseq2 Wald** | **CPM** | `~ sex + disease` | M12 / R6 |
| five-against-five resampling | **edgeR-QL** | **CPM**, within draw | `~ disease` | M14 / R8 |
| paired tissue contrast | **edgeR-QL** | **CPM** (was TMM to 2026-10-19) | `~ patient + tissue (+ tissue:disease)` | M18 / R10 |
| lens replication arm | edgeR-LRT | TMM, theirs | `~ disease` | M20 / R12 |
| classifiers | scikit-learn random forest, all features | log2(CPM + 1) | n/a | M16 / R9 |

One row is still pydeseq2: the simulated power grid. It is not "the primary model
refitted", and M12 recommends dropping it. The five-against-five resampling was moved
onto edgeR-QL (2026-10-12) for exactly this reason; its pydeseq2 result is retained as a
cross-check and the two agree closely.
''')

md('''
## M9. Multiple-testing correction

Benjamini–Hochberg FDR was applied within tissue, across all tested entries, with the
primary endpoint fixed in advance as **the number of miRNAs at BH q < 0.05 per tissue**.
Storey's π₀ — the estimated proportion of true nulls — is reported alongside, because
when nothing survives, π₀ is the more informative statistic: π₀ near 1 says the p-value
distribution is indistinguishable from uniform, i.e. there is no signal to correct *for*,
which is a stronger claim than "no hit reached 0.05".
''')

md('''
## M10. Permutation calibration

The analytic null assumes the tests are independent. They are not: library composition
moves blocks of miRNAs together, so the shuffle-to-shuffle spread in hit count is several
times the binomial width and the theoretical expectation is the wrong reference. The null
was therefore built empirically.

Disease labels were permuted across the samples of one tissue — sex stays attached to its
sample — and the identical pipeline re-run. Dispersion estimation and the model fit are
redone in every run; the expression filter and the size factors are not, being held at the
real labels on purpose so the feature set is constant. 1,000 permutations were run through
edgeR-QL keeping **every p-value of every run**, and a separate 60-shuffle run through the
Wald arm keeping the ten strongest p-values per run.

Three quantities are read off the permutation distribution, and they answer different
questions:

* **the hit-count permutation p**: the fraction of shuffles producing at least as many
  miRNAs at raw P < 0.05 as the real labels did. This is a calibration statistic for the
  run as a whole, not evidence about any individual miRNA.
* **the familywise (best-of-run) p**: the fraction of shuffles whose single best p-value
  is at least as small as our best. This is the right question to ask of a top hit found
  by screening many features.
* **the per-feature permutation p**: for one named miRNA, the fraction of shuffles in
  which *that* miRNA alone reached at least as small a p-value. It ignores the fact that
  the miRNA was selected by screening, and is reported only alongside the familywise
  figure.

All permutation p-values use the Phipson–Smyth correction (k+1)/(n+1), so none is
reported as zero.

**The permutation calibrates the combined contrast, not disease.** Permuting disease
also permutes eyedrop exposure, which is collinear with it.
''')

md('''
## M11. Equivalence testing

A negative result stated as "nothing was significant" conflates absence of evidence with
evidence of absence, and is read by a reviewer as underpowering. The load-bearing
statistic here is therefore equivalence: two one-sided tests (TOST) against a
pre-specified bound Δ, which reject the hypothesis that the effect is *larger* than Δ.

    H0(lower): β ≤ −Δ        H0(upper): β ≥ +Δ
    p_TOST = max[ 1 − Φ((β̂+Δ)/SE),  Φ((β̂−Δ)/SE) ]

Rejecting both at α = 0.05 shows |β| < Δ; equivalently, the 90% confidence interval lies
inside ±Δ. Each miRNA falls into one of four states: *equivalent* (TOST rejects, the
difference test does not), *different* (the reverse), *trivial* (both reject — detectable
but smaller than Δ) and *inconclusive* (neither).

**Δ = 1.0 on the log₂ scale — a two-fold change — was pre-specified** before the tests
were run, and the choice is recorded in `manuscript/notes/provisional_decisions.md`. It
sits inside every effect the lens comparator claims (1.66–2.05 log₂ units) so that those
claims can be excluded, and above our own detection floor. Δ = 1.66, the comparator's
weakest claim, is reported alongside, and the full curve of results against Δ is given in
the appendix so that no single bound can be read as cherry-picked.

The **achieved bound** is also reported per miRNA: |β̂| + crit·SE, the smallest Δ at
which that miRNA would be called equivalent. For the lead edgeR-QL arm `crit` is the t
critical value at the quasi-likelihood residual degrees of freedom, 1.668 to 1.680 in
these data, NOT the normal 1.645; only the asymptotic comparator arms use 1.645. Its distribution answers "what does this
study actually exclude?" without reference to any pre-set bound.

edgeR-QL is the lead arm here as well. edgeR does not report a standard error, but a QL
F-test on a single coefficient has one numerator degree of freedom, so F = t² exactly and
SE = |log₂FC|/√F; this was verified against edgeR's own p-values to 3 × 10⁻¹⁵. The QL
residual degrees of freedom are the proper denominator, so the lead arm is a t-test rather
than an asymptotic normal. The pydeseq2 and NBSR arms use the normal critical value
because their inference is asymptotic.
''')

md('''
## M12. Power

Two calculations, reported together because they disagree in an informative direction.

* **Analytic**: two-sample t-test power over a grid of Cohen's d at the realised group
  sizes. This gives the effect detectable at 80% power.
* **Simulated**: the honest version for count data. Each entry's observed mean and
  dispersion are taken as given, a known fold change is injected into a random half of
  the samples, the model is refitted, and detection at P < 0.05 is recorded. This
  respects the real dispersion, library sizes and filter.

**RECOMMENDATION: drop the simulated grid from the manuscript.** Three reasons.

1. **Its engine is not the primary test.** It refits with a negative-binomial GLM and
   Wald test (pydeseq2) under `~ sex + disease` (`scripts/stage_3/de.py:run`, called from
   `scripts/stage_4/power.py`), not edgeR-QL. Whether edgeR-QL gives the same curve has
   not been tested.
2. **Equivalence already answers the question, on the primary engine.** The achieved
   bound per miRNA (M11) is what this cohort excludes, measured by the test the paper
   reports. The publication plan designated TOST the load-bearing statistic for precisely
   this reason.
3. **Nothing depends on it.** It is not a main-text or appendix figure. In the current
   draft it supports exactly one sentence, the caveat that small effects are not excluded.

That caveat survives the cut and reads better from the bounds: *the median achieved bound
was 0.48 log2 units in lens and 0.57 in tear, so for the typical miRNA effects below
about 1.4-fold are not excluded by this study.* Same claim, primary engine, no
simulation.

The analytic calculation is engine-free (a two-sample t-test power curve at the realised
group sizes) and could be kept if a reviewer asks for a conventional power statement. The
numbers below are left in place pending the decision.

The power ceiling is reported rather than defended. The two calculations disagree about
which tissue is better powered — the analytic one favours tear, which has more samples,
and the simulation favours lens, whose libraries are far less noisy per miRNA — and the
simulation is the operative figure because it uses the realised dispersion. A reviewer
will find the ceiling whether or not we state it first.
''')

md('''
## M13. Stability

Any candidate surviving the screen was asked whether it depends on a handful of patients.
Every refit below uses **edgeR-QL, the same model that nominated the candidate**
(`scripts/stage_4/stability.py`, which previously used pydeseq2 and was changed for this
reason). This is the one resampling analysis in the paper that runs on the primary
engine.

* **Leave-one-out**: refit dropping each sample in turn; the worst p-value obtained is
  reported. A result that depends on one patient is not a result.
* **Random-3**: refit 200 times, each dropping three randomly chosen samples; the
  fraction of draws keeping P < 0.05 is reported, with the p-value spread. This is the
  fragility test used to classify a candidate as stable.
* **Drop-top-3**: refit without the three highest-count samples for that entry. Computed
  and reported, but **not** used to judge stability, because it is structurally biased
  against true effects: the highest expressers come disproportionately from whichever
  group is higher (mean 2.32 of 3 in this cohort against ~1.5 by chance), so removing them
  mechanically shrinks a real difference. Among our candidates, those with all three from
  the higher group moved from median P = 0.006 to 0.18, while those with at most one moved
  from 0.045 to 0.033 — the test punishes an effect for being real.
''')

md('''
## M14. Small-study resampling

The literature this paper responds to is built largely on five-against-five discovery
cohorts. To ask what a cohort of that size yields from data where the full-cohort answer
is known, five XFG and five control samples were drawn at random from our **lens
capsule** cohort, matching the lens comparator's tissue and group sizes,
500 times, keeping each sample's true disease label, and the analysis re-run on each
draw.

**What is ours and what is theirs, precisely.** The draws are analysed with **our** data
and **our** pipeline: our counts, our expression filter, CPM size factors within the
draw, and **the edgeR quasi-likelihood F-test, the same test as the reported differential
expression** (`scripts/stage_4/fig_smallstudy_noise.py` with `smallstudy_edger.R`). What
is taken from the comparator is only the **reporting rule**: their threshold of P < 0.1,
and their weakest reported effect, |log₂FC| ≥ 0.99 (their six effects span 0.99 to 2.05),
as the floor for the match count. This is **not** their pipeline: theirs was edgeR with a
likelihood-ratio test, TMM normalisation, and their own ≥3.7 CPM in ≥50% of samples
filter. Do not describe it as re-running their analysis.

Two further design points that belong in the text:

* **The design drops sex** (`~ disease` rather than `~ sex + disease`). A five-per-group
  draw is frequently single-sex, which makes the adjusted design rank-deficient; dropping
  the term keeps every draw rather than discarding the single-sex ones, which would bias
  the sample toward mixed-sex draws. The comparator did not adjust for sex either. This
  is the one stated difference from the primary model.
* **The filter is held at the full-cohort value**, so every draw tests the same 226
  entries. A real five-per-group study would filter on its own ten samples and test
  fewer. Holding it fixed makes the draws comparable with each other and with the
  full-cohort run, at the cost of not reproducing the multiplicity a real small study
  would face.

Two counts are recorded per draw: how many miRNAs reach P < 0.1, and how many do so with
|log₂FC| ≥ 0.99. A parallel run assigns labels at random within the draw, separating
"what a small sample of this data gives you" from "what a small sample of noise gives
you".

The draws are **not independent studies**: they resample 43 fixed samples, so one patient
appears in many draws and an outlier propagates. The claim is what a small study drawn
from this cohort would report, not a claim about independent replication.

**Engine cross-check.** The whole resampling was also run through a negative-binomial GLM
with Wald test (pydeseq2) on the identical 500 draws, same seeds and same sample lists.
The two agree, so the result is a property of the sample size and not of the test.
''')

md('''
## M15. Application of published criteria

Each prior study's significance criterion, taken from that paper's own Methods, was
applied unchanged to our differential-expression output in both tissues, and the resulting
count compared with the number of miRNAs that study reported. Criteria and provenance are
recorded in `refs/glaucoma_mirna_studies.tsv`, verified against full text where it was
available.

**What transfers is the threshold, not the design.** Tissues, platforms and case
definitions differ between all of these studies and ours, and two of the five were
targeted panels rather than discovery screens — one tested 22 pre-selected miRNAs and
reported none significant, so applying its "P < 0.05" to our 226- and 259-feature screens
is a different test, not a repeat of it. The comparison is a statement about what a
threshold returns from a larger cohort, and nothing stronger.
''')

md('''
## M16. Classification

A classifier was asked the question the DE analysis asks, in a form that does not require
any single miRNA to be significant: can disease group be predicted from the miRNA profile?

Random forests (300 trees, balanced class weights) were fitted per tissue on
log₂(CPM + 1).
All transforms are per-sample — CPM divides a sample by its own total, log₂ is elementwise
— so no information crosses a fold boundary. TMM and z-scoring would leak and were not
used.

Cross-validation is a **single pass of stratified 5-fold**, so each sample is scored
out-of-fold exactly once. The 10-repeat version and the leave-one-out variant are
superseded and are not reported (RP 2026-10-12); the 10-repeat output is archived as
`rf_kfold_10repeat_2026-09-16.tsv`.

Feature handling was varied deliberately, to measure how much apparent performance
feature selection manufactures:

* **all-features**: no selection; the forest selects internally, within each fold.
* **honest**: top-K by t-test, re-selected inside each training split.
* **leaked**: top-K by t-test, selected once on the full matrix, then cross-validated.
  This is the conventional "take the DE hits and classify with them" pipeline.
* **nested RFE**: recursive feature elimination inside each training fold only, subset
  size chosen by inner 3-fold AUC.
* **backward stepwise logistic regression on AIC** and **L1-penalised logistic
  regression**, both with selection strictly in-fold. Stepwise needs a univariate
  pre-screen because the full model (226–259 predictors against n = 43–51) is not
  identifiable; the screen size is swept rather than chosen, each value with its own
  null.

**Every variant is permutation-tested under its own procedure.** This matters most for
the leaked arm: leaking inflates the null as well as the observed value, so a leaked AUC
compared against 0.5 is how "AUC 0.82" headlines are produced from noise. Feature
importances from a full-data fit are descriptive only and are not a biomarker panel.
''')

md('''
## M17. Unsupervised structure

Principal components were computed on log₂ CPM, column-centred, by SVD, over **the same
entry set as the differential expression** (M7), and the clustered heatmap uses that set
as well. One filter and one normalisation govern the whole paper.

**This was not the original design, and the change was tested before it was made.** The
PCA previously used a looser, label-blind rule of its own (≥10 reads in ≥20% of the
samples in the projection), which left the unsupervised section internally inconsistent:
`heatmap.py` had always used the DE filter while the PCA figures did not. Re-running both
with the entry set as the only difference, normalisation being CPM either way:

| | PCA's own filter | DE filter |
|---|---|---|
| joint: entries / PC1 / tissue *d* | 311 / 71.8% / 16.12 | 299 / 72.6% / 16.25 |
| lens: entries / disease *d* / *P* | 258 / 0.03 / 0.92 | 226 / 0.02 / 0.95 |
| tear: entries / disease *d* / *P* | 302 / 0.42 / 0.14 | 259 / 0.42 / 0.14 |
| sample 40A: peer agreement | ρ 0.704, rank 1 of 52 | ρ 0.713, rank 1 of 52 |
| sample 40A: clustering | singleton, k = 3 to 6 | singleton, k = 3 to 6 |

Nothing rests on the choice, including the outlier exclusion, which was the one result
that would have been awkward to re-derive.

**One number did move, in our favour, so state it rather than bank it quietly.** Under the
old filter, Schirmer score in lens reached nominal *P* = 0.045 against PC1, the single
covariate to do so in 22 tests. Under the standardised filter it is *P* = 0.081, and no
covariate reaches *P* < 0.05. The change is a borderline association crossing a threshold,
not a finding appearing or disappearing, and the covariate table should be read that way.

**One argument against it, recorded rather than buried.** The DE filter conditions on the
disease groups, so it is not label-blind, and an unsupervised figure built on it is mildly
circular. The bias runs toward *finding* group structure, since an entry seen in half the
controls and in none of the cases passes the rule, so a null under this filter is the more
conservative statement of the two. The table shows it does not move the answer in any
case. PCA rather than UMAP: at this n with one dominant factor, variance explained is
interpretable and the distances mean something, whereas UMAP's neighbour parameter
manufactures apparent clusters.

The joint projection of both tissues is dominated by the tissue axis, so within-tissue
structure was examined in PCAs **recomputed within each tissue** rather than as a zoom of
the joint one. Group separation on PC1 is quantified as Cohen's d with a two-sample
t-test, and every recorded covariate was tested against PC1 the same way — binary
covariates by d and t-test, continuous ones by Spearman correlation — so that the
covariate panels are comparable with the disease panel.

A clustered heatmap was also produced, z-scoring each miRNA and clustering rows and
columns hierarchically on Euclidean distance with complete linkage. Complete linkage is
the conservative choice: Ward minimises within-cluster variance and so carves tidy blocks
out of noise, which would flatter a figure whose point is that the only real structure is
tissue.
''')

md('''
## M18. Paired lens-versus-tear contrast

The 42 patients who gave both specimens allow the tissue difference to be measured within
patient, so that every between-patient nuisance (age, sex, batch, handling) cancels. Two
models were fitted on these 84 libraries, over the 290 entries passing the M7 filter
evaluated on them:

    ~ patient + tissue                     fitted separately in controls and in XFG
    ~ patient + tissue + tissue:disease    the interaction model

Both use edgeR-QL with robust dispersion and **CPM size factors, matching M7 and the rest of
the reported analyses**. Both scripts used TMM until 2026-10-19; it was changed per RP so that
one normalisation governs the whole paper. The switch moves no conclusion: the interaction is
0 at FDR < 0.05 either way, the two groups' effect sizes correlate at 0.99 either way, and the
tissue main effect goes from 246 to 230 of 290. The raw-P interaction count improves from 16
to 10 against 14.5 expected. TMM was in any case the harder choice to defend here, since it
assumes most entries are unchanged while 230 of 290 differ between the tissues.

Blocking on patient absorbs the disease main effect, since disease is constant within a
patient. That is intended: the question is whether the lens-versus-tear difference itself
*shifts* with disease, which is the interaction term. If it does not, the two disease
groups can be pooled when describing tissue differences, and the licence to pool is
established rather than assumed.

The 10 patients contributing one specimen are excluded from this analysis rather than
assumed to behave like the paired ones.
''')

md('''
## M19. Target-set and functional enrichment

Enrichment was attempted and is reported with its failure modes, because it is the step at
which this kind of study most easily manufactures a result.

Over-representation analysis of pooled predicted targets against a generic gene background
is close to uninterpretable here: abundant miRNAs collectively target most of the genome,
so "enrichment" largely recovers the background. Three constructions were run instead:

* **tissue contrast**: targets of lens-enriched versus tear-enriched miRNAs, against a
  universe of every gene targeted by any miRNA we express. Target sets come from MSigDB
  C3:MIR:MIRDB, with seed-merged set names split so every family member is annotated, and
  a gene counts only if hit by at least two distinct **seed families**: grouping by miRNA
  rather than seed was wrong, because seed-sharing miRNAs get near-identical target lists
  and a single seed can clear a "three independent miRNAs" bar.
* **miRNA-level enrichment** (Godard & van Eyll strategy 3) — each pathway is converted
  into the set of miRNAs that target it and the miRNA list is tested directly, each miRNA
  counted once, so seed families, target-count differences and 3′UTR length cannot inflate
  anything. Run as GSEA over all expressed miRNAs ranked by log₂(lens CPM / tear CPM), and
  as ORA of the top-N most abundant per tissue.
* **annotation-depth-matched null**: the ORA was calibrated against 1,000 random miRNA
  sets drawn from the same universe and matched on annotation depth, because abundant
  miRNAs are better annotated and an unmatched null is too easy to beat.

No enrichment analysis was run on the disease contrast, for the sufficient reason that
there is no gene set to enrich: no miRNA is differentially expressed.
''')

md('''
## M20. Replication arms

Two comparator studies were reproduced from the count matrix onward — their pipeline,
their filter, their test, their threshold — to establish that our specimens resemble the
published description of each tissue before any disease comparison is made. The argument
is "our samples look like theirs, therefore our null is informative", and it must rest on
expression profile, never on agreement about differential expression: using DE
concordance to validate the samples and then reporting DE discordance would be circular.

**Lens capsule — Tomczyk-Socha et al. 2022** (*Genes* 13:582; 5 PEX versus 5 controls).
Reproduced as bowtie2 against miRBase v22 mature, their expression filter of 3.7 CPM in
≥50% of samples, TMM normalisation, edgeR likelihood-ratio test, disease-only design, and
their P < 0.1 threshold. Their cohort is exfoliation syndrome *without* glaucoma and
drop-naive by their own exclusion criteria; ours is XFG and necessarily on IOP-lowering
therapy — so the eyedrop confound is ours alone and cannot be calibrated against theirs.
Disease stage runs the other way and favours detection, since every XFG patient has
exfoliation material in the capsule.

**Tear fluid — Jones et al. 2026** (*Ocul Surf* 40:133–143; 32 healthy adults).
Reproduced as Cutadapt → bowtie1 `-v 0` on 21–24 nt reads, mature only, against their
detection targets: species detected, constitutive (present in ≥50% of samples **and** mean
raw count > 5 — both criteria), present in all subjects, and the five most abundant. Their
stable-miRNA analysis was also reproduced. That paper is descriptive on disease but did
perform differential expression on sex, race and age; two of its race-associated hits are
miR-375-3p and miR-708-5p. Our cohort is ancestry-homogeneous so this cannot confound us,
but the miRNAs that vary most by demography in healthy tears are the ones a disease screen
in tears flags.

Both arms are run **as published**, without our artefact exclusion list, since neither
comparator had one. Family-level comparisons were attempted and are **not** reported:
miRBase families are grouped by hairpin homology and counted per precursor, which
seed-based assignment cannot reproduce, and the apparent deficit is an artefact of the
method rather than a compositional difference.

Our controls are cataract-surgery patients, not healthy adults — matching the lens
comparator's controls but not the tear reference's. This limits the tear comparison to
composition and detection, not group differences.
''')

md('''
## M21. Software

Versions are printed below from the live environment. Analyses run in R are called from
Python with `R_PROFILE_USER=/dev/null` and an explicit `R_LIBS`, because a user-level
`.Rprofile` otherwise prepends an older limma that edgeR 4.x will not load against — a
silent failure worth recording.

Code and analysis outputs are in the project repository; the per-figure scripts are
`scripts/manuscript/fig1.py` … `fig6.py`, and each writes only the figure named in its
docstring.
''')

code(r'''
head('M21  software versions')
import platform, sklearn, scipy, statsmodels, matplotlib
mods = [('python', platform.python_version()), ('numpy', np.__version__),
        ('pandas', pd.__version__), ('scipy', scipy.__version__),
        ('statsmodels', statsmodels.__version__), ('scikit-learn', sklearn.__version__),
        ('matplotlib', matplotlib.__version__)]
try:
    import pydeseq2; mods.append(('pydeseq2', pydeseq2.__version__))
except Exception:
    pass
for n, v in mods:
    print(f'  {n:<14} {v}')
import subprocess
try:
    r = subprocess.run(['Rscript', '-e',
                        'cat(R.version.string, "| edgeR", '
                        'as.character(packageVersion("edgeR")), "| limma", '
                        'as.character(packageVersion("limma")))'],
                       capture_output=True, text=True, timeout=120,
                       env={**os.environ, 'R_PROFILE_USER': '/dev/null',
                            'R_LIBS': '/r_packages'})
    print('  R             ' + (r.stdout.strip() or r.stderr.strip().splitlines()[-1]))
except Exception as e:
    print(f'  R             not queried ({type(e).__name__})')
''')

# results cells

md('''
---
# Part 2 — Results
---

The results are ordered as the argument runs: the specimens match published
descriptions of both tissues (R1); the only structure in them is tissue, not disease
(R2); no miRNA is differentially expressed (R3); the thresholds used in the prior
literature nevertheless return hit counts of the same size as those studies reported
(R4); the permutation, equivalence, power, stability and small-cohort analyses say what
that does and does not license (R5–R8); no classifier separates the groups (R9); and the
tissue difference is large, reproducible and identical in health and disease (R10). One
candidate miRNA survives correction and is treated sceptically (R11).
''')

md('''
## R1. The specimens reproduce published descriptions of both tissues

Before any disease comparison, the data were checked against the published description
of each compartment. This is the validation on which an informative null depends, and it
is deliberately made on expression profile alone.

![Figure 1](../figures/figure1.png)

**Figure 1.** (A) Mean CPM per miRNA in our tear libraries against the published tear
miRNome reference, log–log, with the reference's own core set (detected in ≥50% of
samples in both cohorts) distinguished. (B) The ten miRNAs the lens-capsule comparator
reports as most abundant, as a share of all lens reads in their control group and in
ours.
''')

code(r'''
head('R1  composition agreement')
MIN_DET = 0.5
d = tsv('replication', 'tear_jones_meansd.tsv', branch='CBGM-v22').set_index('Unnamed: 0')
d = d[(d.theirs_mean > 0) & (d.ours_mean > 0)]
x, y = np.log10(d.theirs_mean), np.log10(d.ours_mean)
core = (d.det_theirs >= MIN_DET) & (d.det_ours >= MIN_DET)
say(f'RESULTS -> Mean abundance per miRNA agreed closely with the published tear '
    f'reference: Pearson r = {stats.pearsonr(x, y).statistic:.3f} across all '
    f'{len(d)} miRNAs detected in both cohorts, and r = '
    f'{stats.pearsonr(x[core], y[core]).statistic:.3f} across the '
    f'{int(core.sum())} miRNAs in the reference core set, over roughly seven orders '
    f'of magnitude.')

sc = tsv('replication', 'tear_cbgm.tsv', branch='CBGM-v22')
sc = sc[(sc.policy == 'as-published') & (~sc.target.str.startswith('[BLOCKED]'))]
# cohort labels carry their own n since tear_cbgm.py was fixed on 2026-10-19; do not
# hardcode them, that is what broke this cell when the tear set went from 52 to 51
_cohorts = [(c, 'all tear libraries' if c.startswith('all-') else 'controls only')
            for c in sorted(sc.cohort.unique(), reverse=True)]
for coh, lab in _cohorts:
    s = sc[sc.cohort == coh].set_index('target')
    say(f'RESULTS -> Under their own filters, {lab}: '
        f'{int(s.loc["detected", "ours"])} miRNA species detected against their '
        f'{int(s.loc["detected", "theirs"])}; '
        f'{int(s.loc["constitutive", "ours"])} constitutive against their '
        f'{int(s.loc["constitutive", "theirs"])}; '
        f'{int(s.loc["universal", "ours"])} present in every subject against their '
        f'{int(s.loc["universal", "theirs"])}.')
say('RESULTS -> Four of their five most abundant tear miRNAs occupy the same rank in '
    'our data (let-7a-5p, let-7f-5p, let-7b-5p, let-7g-5p); let-7c-5p is our 8th '
    'rather than their 5th.')
say('CAVEAT -> Their constitutive and universal counts are the two that fall short, and '
    'both are detection thresholds rather than abundance measures: they profiled 32 '
    'healthy adults and ours are cataract-surgery patients, with a different kit. '
    'Detection breadth is the measure most sensitive to both.')

comp = tsv('replication', 'lens_genes2022_composition.tsv',
           branch='PEXlens-v22').set_index('miRNA')
mref = gg2.matrix('PEXlens-v22')
lens_ctrl = [c for c in mref.columns if MD.get('tissue_name', pd.Series()).get(c) == 'lens'
             and MD.disease.get(c) == 'control']
pct = mref[lens_ctrl].sum(axis=1) / mref[lens_ctrl].sum().sum() * 100
print()
say(f'RESULTS -> In lens capsule, miR-184 is the dominant species in both cohorts: '
    f'{pct.get("hsa-miR-184", float("nan")):.1f}% of all reads in our '
    f'{len(lens_ctrl)} controls against the {comp.control_pct.get("hsa-miR-184"):.1f}% '
    f'they report in theirs.')
top5_ours = pct.sort_values(ascending=False).head(5).sum()
say(f'RESULTS -> Their top five account for {comp.control_pct.head(5).sum():.0f}% of '
    f'their reads; our own top five account for {top5_ours:.0f}% of ours.')
disc = [(e, comp.control_pct[e], pct.get(e, 0.0)) for e in comp.index
        if pct.get(e, 0.0) > 0 and (comp.control_pct[e] / max(pct.get(e, 1e-9), 1e-9) >= 3
                                    or pct.get(e, 0.0) / comp.control_pct[e] >= 3)]
for e, th, ou in sorted(disc, key=lambda r: -max(r[1] / max(r[2], 1e-9), r[2] / r[1])):
    f = th / ou if th > ou else ou / th
    say(f'RESULTS -> {e.replace("hsa-", "")} differs {f:.0f}-fold between cohorts '
        f'(theirs {th:.3g}%, ours {ou:.3g}%).')
say('CAVEAT -> Ligation bias is miRNA-specific and well documented, and they used a '
    'Lexogen kit against our QIAseq. That is the leading explanation for the '
    'miR-204-5p and miR-486-5p discrepancies, which are the two largest and are '
    'reported as unexplained rather than reconciled.')
''')

md('''
## R2. The only structure in the data is tissue

![Figure 2](../figures/figure2.png)

**Figure 2.** (A) PCA of all libraries, coloured by tissue. (B, C) PCA recomputed within
lens and within tear, coloured by disease group. Cohen's d on PC1 is given per panel.
''')

code(r'''
head('R2  unsupervised structure')
Pj, varj, nfj = pca.pca(MX, keep=KEEP['joint'])
is_lens = (MD.tissue_name == 'lens').values
dj = pca.d_between(Pj.PC1[is_lens], Pj.PC1[~is_lens])
say(f'RESULTS -> On {nfj} filtered miRNAs, PC1 explains {100*varj[0]:.1f}% of total '
    f'variance and separates the two tissues completely (Cohen\'s d = {dj:.1f}); PC2 '
    f'explains {100*varj[1]:.1f}%. No lens library falls among the tear libraries or '
    f'the reverse.')

for t in ('lens', 'tear'):
    Pt, vt, nf = pca.pca(MX[TISSUE[t]], keep=KEEP[t])
    sub = MD.loc[TISSUE[t]]
    g = [Pt.PC1[sub.index[sub.disease == k]] for k in ('control', 'diseased')]
    say(f'RESULTS -> Within {t} ({nf} miRNAs, PC1 {100*vt[0]:.1f}%), the disease groups '
        f'do not separate: d = {pca.d_between(*g):.2f}, '
        f'P = {stats.ttest_ind(*g).pvalue:.2f}.')
say('RESULTS -> The contrast between a d of 16 for tissue and a d below 0.5 for disease '
    'in the same projection is the single most compact statement of the negative result, '
    'and it is made without any model or significance threshold.')
''')

md('''
### R2b. [APPENDIX] No recorded covariate structures the within-tissue projection

Every clinical covariate was tested against PC1 within each tissue, on the same
projection as Figure 2B–C, so that the disease panel and the covariate panels are
directly comparable. This answers the reviewer question "is something else driving your
PC1?" before it is asked.

*Appendix figure to be drawn from this cell: PC1/PC2 within each tissue, one small panel
per covariate, disease panel repeated for reference.*
''')

code(r'''
head('R2b  covariates against PC1, within tissue')
BIN = ['disease', 'sex', 'diabetes', 'hypertension', 'dementia', 'smoke', 'vitamin',
       'family_history']
CONT = ['age', 'schirmer', 'cigs_per_day']
rows = []
for t in ('lens', 'tear'):
    Pt, vt, nf = pca.pca(MX[TISSUE[t]], keep=KEEP[t])
    sub = MD.loc[TISSUE[t]]
    for cov in BIN:
        v = sub[cov]
        lv = [k for k in v.dropna().unique()]
        if len(lv) != 2:
            continue
        g = [Pt.PC1[v[v == k].index] for k in lv]
        rows.append(dict(tissue=t, covariate=cov, kind='binary',
                         n=f'{len(g[0])}/{len(g[1])}',
                         stat=round(pca.d_between(*g), 2),
                         p=round(stats.ttest_ind(*g).pvalue, 3)))
    for cov in CONT:
        v = pd.to_numeric(sub[cov], errors='coerce')
        ok = v.notna()
        if ok.sum() < 10:
            continue
        r = stats.spearmanr(Pt.PC1[ok.values], v[ok])
        rows.append(dict(tissue=t, covariate=cov, kind='rho', n=int(ok.sum()),
                         stat=round(r.statistic, 2), p=round(r.pvalue, 3)))
    tot = MX[TISSUE[t]].sum()
    r = stats.spearmanr(Pt.PC1, np.log10(tot))
    rows.append(dict(tissue=t, covariate='library depth', kind='rho', n=len(tot),
                     stat=round(r.statistic, 2), p=round(r.pvalue, 4)))
cv = pd.DataFrame(rows)
print(cv.to_string(index=False))
print()
sig = cv[(cv.p < .05) & (cv.covariate != 'library depth')]
say(f'RESULTS -> Of {len(cv) - 2} covariate tests across the two tissues, '
    f'{len(sig)} reached nominal P < 0.05 '
    f'({", ".join(f"{r.covariate} in {r.tissue}" for r in sig.itertuples()) or "none"}), '
    f'against {0.05*(len(cv)-2):.1f} expected by chance. No covariate approaches the '
    f'separation that tissue produces, and none is a plausible explanation for the '
    f'disease null.')
dep = cv[cv.covariate == 'library depth']
say('RESULTS -> PC1 within each tissue is strongly associated with library depth: '
    + ', '.join(f'{r.tissue} rho = {r.stat:+.2f}' for r in dep.itertuples()) + '.')
say('CAVEAT -> That depth association is the honest finding of this panel: the leading '
    'within-tissue axis tracks library purity, not biology. It is the reason the disease '
    'test is a model on counts with size factors rather than a distance in PCA space, '
    'and it is why the depth-check appendix figure is included.')
print()
print('appendix figures already drawn:')
for f in ('fig_pca_disease.png', 'fig_pca_tissue_disease.png',
          'fig_pca_depth_check.png', 'fig_pca_tissue_depth.png'):
    print(f'  plots/branches/{ARM}/stage_2/{f}')
# the heatmap is Supplementary Figure S2 and lives with the manuscript figures
print('  manuscript/figures/figureS2_clustering.png')
''')

md('''
### R2c. [APPENDIX] Hierarchical clustering recovers tissue and nothing else

![Clustered heatmap](../figures/figureS2_clustering.png)

**Appendix figure.** Unsupervised clustering of z-scored log₂ CPM, complete linkage on
Euclidean distance, with tissue and disease annotated above the columns. Cut into four
groups the clustering separates the tissues perfectly; the disease labels never separate
at any cut from k = 2 to k = 6, under any linkage. The within-tissue column ordering
tracks library purity.
''')

md('''
## R3. No miRNA is differentially expressed in either tissue

![Figure 3](../figures/figure3.png)

**Figure 3.** (A, B) Volcano plots for lens capsule and tear fluid, edgeR
quasi-likelihood F-test on CPM-normalised counts with the design `~ sex + disease`. The
dashed line is the rank-1 Benjamini–Hochberg critical value q/n, where BH and Bonferroni
coincide. (C) miR-708-5p in tear by group, every sample shown.
''')

code(r'''
head('R3  differential expression')
de = tsv('stage_3', 'de_edger.tsv')
de = de[(de.norm == 'CPM') & (de.kind == 'relative')]
cs = tsv('stage_4', 'correction_summary.tsv')
cs = cs[(cs.method == 'edgeR-QL') & (cs.norm == 'CPM')].set_index('tissue')
res = {}
for t in ('lens', 'tear'):
    s = de[de.tissue == t]
    q = multipletests(s.ql_p.values, method='fdr_bh')[1]
    res[t] = dict(n=len(s), p05=int((s.ql_p < .05).sum()), exp=.05 * len(s),
                  minp=s.ql_p.min(), minq=q.min(), bh05=int((q < .05).sum()),
                  bh10=int((q < .10).sum()), pi0=cs.pi0.get(t))
    r = res[t]
    say(f'RESULTS -> {t.capitalize()}: {r["n"]} miRNAs passed the expression filter and '
        f'were tested. {r["p05"]} reached raw P < 0.05 against {r["exp"]:.1f} expected '
        f'by chance. The smallest p-value was {r["minp"]:.1e}, giving a '
        f'Benjamini-Hochberg q of {r["minq"]:.3f}; '
        + (f'{r["bh05"]} miRNA survived q < 0.05 and {r["bh10"]} survived q < 0.10.'
           if r['bh05'] else 'nothing survived correction at q < 0.05 or q < 0.10.')
        + f' Storey pi0 = {r["pi0"]:.3f}.')
say(f'RESULTS -> In lens the estimated proportion of true nulls is '
    f'{res["lens"]["pi0"]:.2f} - the p-value distribution is indistinguishable from '
    f'uniform, so there is no signal present to correct for. In tear it is '
    f'{res["tear"]["pi0"]:.2f}, leaving room for a small number of real effects, and one '
    f'candidate does survive correction (R11).')
print()
for t in ('lens', 'tear'):
    print(f'  {t} - five smallest p-values')
    print(de[de.tissue == t].nsmallest(5, 'ql_p')[['entry', 'ql_log2FC', 'ql_p']]
          .assign(entry=lambda d: d.entry.str.replace('hsa-', ''))
          .to_string(index=False, float_format=lambda v: f'{v:.4g}'))
''')

md('''
### R3b. [APPENDIX] The null holds under all four models

The lead method is edgeR's quasi-likelihood F-test. Three further models were fitted on
the same counts, filter, design and size factors, and the normalisation family was varied
within each. The appendix reports the full grid; the point of it is that no combination
produces a different answer.
''')

code(r'''
head('R3b  every model x normalisation combination')
allc = tsv('stage_4', 'correction_summary.tsv')
piv = (allc.pivot_table(index=['method', 'norm', 'kind'], columns='tissue',
                        values=['tested', 'p05', 'BH05', 'min_BH'], aggfunc='first')
           .round(3).sort_index())
print(piv.to_string())
print()
nz = allc[allc.BH05 > 0]
say(f'RESULTS -> {len(allc)} combinations of model, normalisation and normalisation '
    f'family were run across the two tissues. {len(nz)} produced any miRNA at '
    f'BH q < 0.05.')
for r in nz.itertuples():
    say(f'RESULTS -> {r.method} / {r.norm} / {r.kind} / {r.tissue}: {int(r.BH05)} at '
        f'q < 0.05 (smallest q = {r.min_BH:.3g}).')
say('CAVEAT -> NBSR is the outlier family here and the reason matters. It models the '
    'library as compositional, so a global shift in how much miRNA a specimen yields is '
    'absorbed into the per-sample composition and re-expressed as many small per-miRNA '
    'effects. Its extra hits are a property of the parameterisation, not a second '
    'opinion about disease, which is why the lead method is edgeR-QL and NBSR is '
    'reported alongside rather than merged.')
print()
print('appendix figures already drawn:')
for f in ('fig_volcano_edger.png', 'fig_volcano_nbsr.png', 'fig_volcano_secondary.png',
          'fig_norm_shift.png', 'fig_pvalue_distribution.png'):
    print(f'  plots/branches/{ARM}/stage_3/{f}')
''')

md('''
## R4. Published thresholds return findings of the published size from these data

This is the central result. Each prior study's own significance criterion was applied
unchanged to our output in both tissues and compared with what that study reported.

![Figure 4](../figures/figure4.png)

**Figure 4.** (A) Each study's published criterion applied to our lens and tear data,
against the number of miRNAs that study reported. (B) What survives correction in our
data: raw P < 0.05, the median across 1,000 label permutations, and the number at BH
q < 0.05. The middle bar is the permutation median rather than 0.05 x n: the tests are not
independent, so the binomial figure is the wrong null (changed 2026-10-19).
Thresholds transfer; designs do not — see the caveat below.
''')

code(r'''
head('R4  published criteria applied to this cohort')
# Designs are given per contrast, matching Table 1: a study that reported several
# contrasts contributes its exfoliation contrast, not its pooled case count.
# Drewry's criterion has THREE conditions and all three are applied, as in fig4.py.
# The third cannot bind here (our own filter is far stricter) but the table must
# state the criterion as the source states it.
CRIT = [
    ('Tomczyk-Socha 2022', 'P < 0.1', lambda s: s.ql_p < .10, 6, 'lens capsule, 5 v 5'),
    ('Rao 2020', 'P < 0.05', lambda s: s.ql_p < .05, 12,
     'peripheral blood, 3 v 3 (XFG against XFS with ocular hypertension)'),
    ('Kosior-Jarecka 2021', 'P < 0.05', lambda s: s.ql_p < .05, 0,
     'aqueous humour, 42 v 36, 22-miRNA panel'),
    ('Drewry 2018', '|log2FC| >= 0.6 and P <= 0.05 and detected in >= 5 samples',
     lambda s: (s.ql_p <= .05) & (s.ql_log2FC.abs() >= .6) & (s.n_det >= 5), 5,
     'aqueous humour, 12 POAG, 12 XFG v 11'),
    ('Cho 2022', 'fold change > 2 and P < 0.05',
     lambda s: (s.ql_p < .05) & (s.ql_log2FC.abs() > 1.), 12,
     'aqueous humour, 11 XFG, 8 NTG v 7'),
    # the fold-change half of Czop's criterion is non-binding (see fig4.py)
    ('Czop 2023', 'P < 0.01', lambda s: s.ql_p < .01, 20, 'aqueous humour, 9 v 9'),
]
sub = {}
for t in ('lens', 'tear'):
    cols = [c for c in MX.columns if MD.loc[c, 'tissue_name'] == t]
    ndet = (MX[cols] > 0).sum(axis=1)
    x = de[de.tissue == t].copy()
    x['n_det'] = x.entry.map(ndet).fillna(0)
    sub[t] = x
rows = [dict(study=n, criterion=c, design=g, reported=th,
             lens=int(f(sub['lens']).sum()), tear=int(f(sub['tear']).sum()))
        for n, c, f, th, g in CRIT]
cr = pd.DataFrame(rows)
print(cr.to_string(index=False))
print()
for r in cr.itertuples():
    say(f'RESULTS -> {r.study} reported {r.reported} miRNAs from {r.design} using '
        f'{r.criterion}. The same criterion applied here returns {r.lens} miRNAs in lens '
        f'and {r.tear} in tear.')
pm = tsv('stage_4', 'perm_edger_summary.tsv').set_index('tissue')
say(f'RESULTS -> None of these counts survives multiple-testing correction: '
    f'{res["lens"]["bh05"]} miRNA in lens and {res["tear"]["bh05"]} in tear at BH '
    f'q < 0.05, against {res["lens"]["p05"]} and {res["tear"]["p05"]} at raw P < 0.05 '
    f'and permutation medians of {pm.loc["lens","null_median"]:.0f} and '
    f'{pm.loc["tear","null_median"]:.0f}. The binomial figures (0.05 x n) would be '
    f'{res["lens"]["exp"]:.1f} and {res["tear"]["exp"]:.1f}, but the tests are not '
    f'independent so that is the wrong null.')
say('CAVEAT -> This is not a like-for-like repeat of any of these studies. Tissue, '
    'platform and case definition differ in every case, and two of the five were '
    'targeted panels rather than discovery screens - Kosior-Jarecka tested 22 '
    'pre-selected miRNAs and reported none significant, so applying P < 0.05 to a 226- '
    'or 259-feature screen is a different test. What transfers is the threshold, and the '
    'claim is correspondingly narrow: thresholds of this kind return counts of this size '
    'from a cohort roughly four to five times the five-versus-five discovery cohorts, in '
    'which nothing survives correction. The range is not uniform: Kosior-Jarecka at 78 '
    'patients is LARGER than either of our tissues, and Drewry only 1.9 to 2.2 times '
    'smaller.')
''')

md('''
## R5. [APPENDIX] The permutation null is wider than the analytic one, and the observed counts sit inside it

The analytic expectation assumes independent tests. Shuffling the disease labels and
re-running the whole pipeline 1,000 times shows how wrong that assumption is here, and
puts the observed hit counts in their proper reference distribution.

![Null calibration](../../plots/branches/BEA-dedup/stage_4/fig_null_calibration.png)

**Appendix figure.** Top: p-value histograms per tissue with the band of shuffled runs
behind the observed bars, and a Kolmogorov–Smirnov test against uniform. Bottom: the
distribution of hit counts at raw P < 0.05 across 1,000 label permutations, with the
observed count marked.
''')

code(r'''
head('R5  permutation calibration')
pe = tsv('stage_4', 'perm_edger_summary.tsv')
for r in pe.itertuples():
    say(f'RESULTS -> {r.tissue.capitalize()}: {int(r.real_p05)} miRNAs at raw P < 0.05 '
        f'on the true labels. Across {int(r.n_perm)} label permutations the median was '
        f'{r.null_median:.0f} (IQR {r.null_q25:.0f}-{r.null_q75:.0f}, 90th percentile '
        f'{r.null_q90:.0f}), so the observed count has a permutation p of {r.perm_p:.2f} '
        f'- entirely unremarkable.')
    say(f'RESULTS -> The analytic expectation for {r.tissue} is {r.theory_p05:.1f} hits '
        f'with a binomial SD of {r.binomial_sd:.1f}; the permutation SD is '
        f'{r.null_sd:.1f}, {r.null_sd/r.binomial_sd:.1f} times wider. Library '
        f'composition moves blocks of miRNAs together, so the independence assumption '
        f'behind the analytic null does not hold and the permutation distribution is the '
        f'correct reference.')
print()
ps = tsv('stage_4', 'permutation_summary.tsv')
print(ps[ps.norm == 'CPM'].to_string(index=False))
say('RESULTS -> The independent 60-shuffle run through the Wald arm agrees: observed '
    'counts at P < 0.05, P < 0.01 and BH < 0.10 all fall at or below the shuffle median, '
    'with no permutation p below 0.27.')
say('CAVEAT -> Because disease and eyedrop exposure are collinear, the permutation '
    'calibrates the combined contrast. It cannot say anything about disease with '
    'treatment held constant, and no analysis in this paper can.')
''')

md('''
## R6. [APPENDIX] What this cohort can exclude

A null result is only informative to the extent that it bounds the effect. Equivalence
testing against a pre-specified bound makes that bound explicit and falsifiable.

![Equivalence field](../../plots/branches/BEA-dedup/stage_4/fig_equivalence_field.png)

**Appendix figure.** Every tested miRNA on the x-axis, ordered by the effect size this
study can exclude, with the achieved bound on the y-axis. The shaded fields separate the
effects ruled out from those still admissible, per tissue, with the effect sizes reported
by the lens comparator plotted for comparison.
''')

code(r'''
head('R6  equivalence and power')
eq = tsv('stage_4', 'equivalence_summary.tsv')
eq = eq[(eq.method == 'edgeR-QL') & (eq.norm == 'CPM')]
for r in eq.itertuples():
    say(f'RESULTS -> {r.tissue.capitalize()}: of {int(r.n)} tested miRNAs, '
        f'{int(r.equiv_1)} ({100*r.equiv_1/r.n:.0f}%) are statistically equivalent at '
        f'the pre-specified bound of a two-fold change - their 90% confidence intervals '
        f'lie entirely inside +/-1 log2 unit. {int(r.diff_1)} are larger than the bound '
        f'and {int(r.incon_1)} are inconclusive. At the comparator\'s weakest claimed '
        f'effect (1.66 log2 units) {int(r.equiv_1p66)} of {int(r.n)} '
        f'({100*r.equiv_1p66/r.n:.0f}%) are excluded.')
    say(f'RESULTS -> The median achievable bound in {r.tissue} is {r.median_bound:.2f} '
        f'log2 units ({2**r.median_bound:.2f}-fold), and 90% of miRNAs are bounded below '
        f'{r.q90_bound:.2f} ({2**r.q90_bound:.2f}-fold).')
print()
pw = tsv('stage_4', 'power.tsv')
an = pw[pw.kind == 'analytic']
for t in ('lens', 'tear'):
    s = an[an.tissue == t].sort_values('d')
    d80 = s[s.power >= .8].d.min()
    say(f'RESULTS -> {t.capitalize()}: analytic power reaches 80% at Cohen\'s '
        f'd = {d80:.2f}; power at d = 0.5 is {100*s.set_index("d").power.get(0.5, np.nan):.0f}%.')
sim = pw[pw.kind == 'simulated']
print()
print(sim.pivot_table(index='d', columns='tissue', values='power')
         .rename(lambda v: f'{2**v:.1f}x').round(3).to_string())
say('RESULTS -> The count-based simulation, which injects a known fold change and '
    'refits the real model, gives '
    + '; '.join(f'{t} {100*sim[(sim.tissue==t) & (sim.d.round(3)==1.0)].power.iloc[0]:.0f}% '
                f'at a two-fold change' for t in ('lens', 'tear')) + '.')
say('CAVEAT -> Lens is the better-powered arm in the simulation and tear the better in '
    'the analytic calculation, because tear libraries are noisier per miRNA while being '
    'more numerous. Report the simulation as the operative figure, and report plainly '
    'that neither arm reaches 80% power below a roughly two-fold change. Effects smaller '
    'than that are not excluded by this study.')
print()
# Compare the comparator's claims against OUR PRIMARY analysis (edgeR-QL, CPM, the
# Section 2.6 filter). This cell previously read a PEXlens-v22 table that was pydeseq2 on
# median-of-ratios with a permissive second fit for the two miRNAs failing the comparator's
# own filter, so it reported 6 testable claims where our analysis tests 5. That table was
# rebuilt on this arm and retired 2026-10-21; see scripts/replication/figs_genes2022.py.
theirs = pd.read_csv(REFS / 'genes2022_lens_de.tsv', sep='\t')
eql = tsv('stage_4', 'equivalence.tsv')
eql = eql[(eql.tissue == 'lens') & (eql.method == 'edgeR-QL')
          & (eql['norm'] == 'CPM')].set_index('entry')
rows = []
for _, r in theirs.iterrows():
    e = r.miRNA
    if e in eql.index:
        q = eql.loc[e]
        rows.append(dict(miRNA=e.replace('hsa-', ''), theirs=round(r.logFC, 3),
                         ours=round(q.log2FC, 3), lo=round(q.ci_lo, 3),
                         hi=round(q.ci_hi, 3),
                         outside=bool(r.logFC < q.ci_lo or r.logFC > q.ci_hi)))
    else:
        rows.append(dict(miRNA=e.replace('hsa-', ''), theirs=round(r.logFC, 3),
                         ours=None, lo=None, hi=None, outside=None))
cmp_ = pd.DataFrame(rows)
print(cmp_.to_string(index=False))
n_out = int(cmp_.outside.fillna(False).sum())
n_untested = int(cmp_.ours.isna().sum())
say(f'RESULTS -> Of the {len(cmp_)} effects the lens comparator reports, {n_out} lie '
    f'outside our 90% confidence interval for the same miRNA in the same tissue, so '
    f'their claimed effect is excluded by our data. '
    + (f'The remaining {n_untested} ('
       f'{", ".join(cmp_.miRNA[cmp_.ours.isna()])}) did not pass our expression filter '
       f'and was not tested.' if n_untested else ''))
''')

md('''
## R7. [APPENDIX] Candidates are fragile to single patients

Every miRNA reaching raw P < 0.05 was refitted leaving out each sample in turn, and
refitted 200 times dropping three random samples.

![Stability](../../plots/presentation/11_de_stability.png)

**Appendix figure.** Per-candidate p-value under the full fit, the worst leave-one-out
fit, and the spread across random three-sample deletions.
''')

code(r'''
head('R7  stability of candidates')
st = tsv('stage_4', 'stability.tsv')
for t in ('lens', 'tear'):
    s = st[st.tissue == t]
    # stability.py:144 defines `stable` as max p across LEAVE-ONE-OUT refits < 0.05,
    # NOT as a threshold on the random-3 proportion. An earlier version of this cell
    # described the random-3 rule while printing the leave-one-out count, which put a
    # wrong criterion into the manuscript. Do not reintroduce an 80% threshold here.
    say(f'RESULTS -> {t.capitalize()}: of {len(s)} candidates at raw P < 0.05, '
        f'{int(s.stable.sum())} remained below P < 0.05 under every single-sample '
        f'deletion and {len(s) - int(s.stable.sum())} did not. Across 200 refits each '
        f'dropping three random samples, the proportion retaining P < 0.05 ranged from '
        f'{s.rand_frac_sig.min():.0%} to {s.rand_frac_sig.max():.0%} '
        f'(median {s.rand_frac_sig.median():.0%}).')
print()
print(st.assign(entry=lambda d: d.entry.str.replace('hsa-', ''))[
        ['tissue', 'entry', 'p_full', 'p_loo_worst', 'rand_frac_sig', 'stable']]
      .to_string(index=False, float_format=lambda v: f'{v:.4g}'))
say('CAVEAT -> Drop-top-3 is computed and tabulated but is not used to judge stability. '
    'It removes the three highest-expressing samples for an entry, and those come '
    'disproportionately from whichever group is higher, so the test shrinks real effects '
    'by construction. The leave-one-out and random-3 results are the ones to report.')
''')

md('''
## R8. [APPENDIX] A five-against-five cohort drawn from these data reproduces the published findings

The discovery cohorts in this literature are frequently five per group. Drawing five XFG
and five control samples at random from our own **lens capsule** data, 500 times, and
re-running the analysis through the primary test answers what a cohort of that size
yields from data in which the full-cohort answer is known to be null. Tissue, group sizes, significance threshold and
effect-size floor match the lens comparator. The quantification, filter, normalisation
and test are ours, not theirs (M14), so this is a statement about what n = 5 per group
does to data like ours, and not a re-run of their study.

![Small-study resampling](../../plots/branches/BEA-dedup/stage_4/fig_smallstudy_5v5.png)

**Appendix figure.** Distribution across 500 draws of the number of miRNAs reaching
P < 0.1, and the number doing so with an effect at least as large as the lens
comparator's, with the comparator's published count marked.
''')

code(r'''
head('R8  five-against-five resampling')
ss = tsv('stage_4', 'smallstudy_5v5.tsv')
say(f'RESULTS -> Across {len(ss)} random draws of 5 XFG against 5 controls, the median '
    f'draw yielded {ss.n_p10.median():.0f} miRNAs at P < 0.1 (IQR '
    f'{ss.n_p10.quantile(.25):.0f}-{ss.n_p10.quantile(.75):.0f}, range '
    f'{ss.n_p10.min():.0f}-{ss.n_p10.max():.0f}) and {ss.n_p05.median():.0f} at '
    f'P < 0.05.')
say(f'RESULTS -> The lens comparator reported 6 miRNAs at P < 0.1 from 5 against 5. '
    f'{100*(ss.n_p10 >= 6).mean():.0f}% of our draws produced at least that many, and '
    f'{100*(ss.n_match >= 6).mean():.0f}% produced at least 6 with an effect at least as '
    f'large as the weakest they reported (|log2FC| >= 0.99).')
py = tsv('stage_4', 'smallstudy_5v5_pydeseq2.tsv')
mg = ss.merge(py, on='seed', suffixes=('_ed', '_py'))
say(f'RESULTS -> Repeating the whole resampling with a negative-binomial Wald test on '
    f'the identical {len(mg)} draws gives a median of {mg.n_p10_py.median():.0f} hits '
    f'against {mg.n_p10_ed.median():.0f}, per-draw counts correlated at r = '
    f'{mg.n_p10_ed.corr(mg.n_p10_py):.3f}, and the same '
    f'{100*((py.n_match >= 6) & (py.n_p05 >= 1)).mean():.0f}% of draws reproducing the '
    f'published profile. The result is a property of the sample size, not of the test.')
say('RESULTS -> The draws use the real disease labels throughout. A finding of the '
    'published size and the published effect magnitude is therefore the typical outcome '
    'of sampling ten patients from this cohort - it does not require noise to be '
    'introduced, only the sample to be small.')
say('CAVEAT -> Do not write that the draws come from a cohort "in which nothing is '
    'differentially expressed". That is not licensed: what the full cohort shows is '
    'that nothing is DETECTED at BH q < 0.05 in lens, and that effects larger than '
    'two-fold are excluded for 94% of lens miRNAs. Say detected, or say bounded, not '
    'absent.')
say('CAVEAT -> This resamples our cohort, not theirs. Tissue, group sizes, threshold and '
    'effect-size floor match the lens comparator; the quantification, filter, '
    'normalisation, test and design are ours, as are the platform, kit, reference '
    'release and case definition. The test is edgeR-QL, matching the primary analysis; '
    'the design drops sex, which is the one stated difference - see M14. It is a '
    'statement about what n = 5 per group does to data like ours, which is the claim '
    'the paper needs and no more than that.')
''')

md('''
## R9. No classifier separates the disease groups

A classifier asks the same question without requiring any single miRNA to be significant.

![Figure 5](../figures/figure5.png)

**Figure 5.** (A, B) ROC from repeated stratified 5-fold cross-validation in lens and
tear, with a stratified bootstrap band over patients. (C, D) Random-forest Gini
importance for every miRNA, against the 5th–95th percentile envelope of 40 fits on
shuffled labels.
''')

code(r'''
head('R9  classification')
kf = tsv('stage_5', 'rf_kfold.tsv')
for r in kf[kf.variant == 'all-features'].itertuples():
    say(f'RESULTS -> {r.tissue.capitalize()}: repeated stratified 5-fold '
        f'cross-validation of a random forest on all {len(sub[r.tissue])} miRNAs gave '
        f'AUC {r.auc:.3f} (SD {r.auc_sd:.3f} across {int(r.n_repeat)} repeats), against '
        f'a permutation null with median {r.null_median:.3f} and 90th percentile '
        f'{r.null_p90:.3f}; permutation p = {r.perm_p:.2f}.')
say('RESULTS -> Neither tissue separates. The observed AUC sits inside the null '
    'distribution in both, and no feature exceeds the shuffled-label importance '
    'envelope.')
print()
print(kf.to_string(index=False))
print()
say(f'RESULTS -> The leaked variant, in which the top 20 features are chosen once on '
    f'the full matrix and then cross-validated, reports AUC '
    f'{kf[(kf.tissue=="lens") & (kf.variant=="leaked")].auc.iloc[0]:.3f} in lens - a gain '
    f'of {kf[(kf.tissue=="lens") & (kf.variant=="leaked")].auc.iloc[0] - kf[(kf.tissue=="lens") & (kf.variant=="honest")].auc.iloc[0]:+.3f} '
    f'over the honest variant, which re-selects inside each fold. Its permutation null '
    f'shifts by almost exactly the same amount '
    f'({kf[(kf.tissue=="lens") & (kf.variant=="honest")].null_median.iloc[0]:.3f} to '
    f'{kf[(kf.tissue=="lens") & (kf.variant=="leaked")].null_median.iloc[0]:.3f}), so the '
    f'permutation p is unchanged. The entire apparent performance of the conventional '
    f'pipeline is leakage, and it is only visible because the null was computed under the '
    f'leaking procedure.')
''')

md('''
### R9b. [APPENDIX] Thirteen classifier configurations per tissue, one nominally significant

Beyond the three 5-fold variants, ten further configurations were run per tissue:
leave-one-out cross-validation, nested recursive feature elimination, explicit
honest-versus-leaked feature selection at k = 20, backward stepwise logistic regression
on AIC across five pre-screen sizes, and L1-penalised logistic regression. All selection
is strictly in-fold and each configuration carries its own permutation null.

The table is the appendix's own argument: it is a complete account of what was tried,
including the configuration that worked.

![Classifier leakage](../../plots/branches/BEA-dedup/stage_5/fig_selection_bias.png)
![Logistic regression](../../plots/branches/BEA-dedup/stage_5/fig_logit.png)
''')

code(r'''
head('R9b  all classifier configurations')
cfg = []
for f, tag in (('rf_loocv.tsv', 'RF, leave-one-out'), ('rf_rfe.tsv', 'RF, nested RFE')):
    d_ = tsv('stage_5', f)
    nullcol = 'null_median' if 'null_median' in d_ else 'null_median_auc'
    for r in d_.itertuples():
        cfg.append((r.tissue, tag, r.auc, getattr(r, nullcol), r.perm_p))
for r in tsv('stage_5', 'rf_selection_bias.tsv').itertuples():
    cfg.append((r.tissue, f'RF, {r.selection} selection k={int(r.k)}', r.auc,
                r.null_median, r.perm_p))
for r in tsv('stage_5', 'rf_kfold.tsv').itertuples():
    cfg.append((r.tissue, f'RF, 5-fold, {r.variant}', r.auc, r.null_median, r.perm_p))
for r in tsv('stage_5', 'logit_stepaic.tsv').itertuples():
    nm = 'logistic, L1' if r.arm == 'lasso' else f'logistic, stepAIC screen={int(r.screen_k)}'
    cfg.append((r.tissue, nm, r.auc, r.null_median, r.perm_p))
cfg = pd.DataFrame(cfg, columns=['tissue', 'configuration', 'auc', 'null_median',
                                 'perm_p']).round(4)
print(cfg.sort_values(['tissue', 'perm_p']).to_string(index=False))
print()
hits = cfg[cfg.perm_p < .05]
say(f'RESULTS -> {len(cfg)} classifier configurations were run across the two tissues. '
    f'{len(hits)} reached a permutation p below 0.05, against {0.05*len(cfg):.1f} '
    f'expected if none of them carried signal.')
for r in hits.itertuples():
    say(f'RESULTS -> The exception is {r.configuration} in {r.tissue}: AUC {r.auc:.3f} '
        f'against a null median of {r.null_median:.3f}, permutation p = {r.perm_p:.3f}. '
        f'It is reported because it is the kind of result this paper is about - one '
        f'configuration out of {len(cfg)}, not replicated by the other '
        f'{len(cfg)-len(hits)}, not supported by any differentially expressed miRNA, and '
        f'exactly what searching over model families produces from null data.')
say('CAVEAT -> Stepwise logistic regression failed to converge in some folds at the '
    'larger pre-screen sizes (43 of 43 folds in lens at screen = 25, 29 of 51 in tear), '
    'because the model is separable. Those are counted and reported rather than dropped, '
    'and the configurations concerned carry no weight.')
''')

md('''
## R10. The two tissues differ profoundly, and identically in health and disease

![Figure 6](../figures/figure6.png)

**Figure 6.** (A) miRNAs passing the expression filter in lens, in tear, or in both. (B)
The paired lens-versus-tear contrast fitted separately in controls and in XFG, restricted
to the 42 patients who gave both specimens.
''')

code(r'''
head('R10  paired tissue contrast')
ts = tsv('stage_6', 'tissue_split_by_disease.tsv')
it = tsv('stage_6', 'tissue_disease_interaction.tsv')
r_ = stats.pearsonr(ts.lfc_ctrl, ts.lfc_xfg).statistic
say(f'RESULTS -> Fitted within patient on the 42 patients giving both specimens, '
    f'{int((ts.fdr_ctrl < .05).sum())} of {len(ts)} miRNAs separate the tissues in '
    f'controls and {int((ts.fdr_xfg < .05).sum())} in XFG, at FDR < 0.05. The two sets '
    f'of effect sizes agree at r = {r_:.3f}.')
say(f'RESULTS -> In the interaction model (~ patient + tissue + tissue:disease), '
    f'{int((it.inter_FDR < .05).sum())} of {len(it)} miRNAs show a tissue-by-disease '
    f'interaction at FDR < 0.05, and {int((it.inter_p < .05).sum())} at raw P < 0.05 '
    f'against {0.05*len(it):.1f} expected by chance. The tissue main effect in the same '
    f'model is significant for {int((it.tissue_FDR < .05).sum())} miRNAs.')
say('RESULTS -> The lens-versus-tear difference is therefore large, highly significant, '
    'and the same in both disease groups. This licenses pooling the groups when '
    'describing tissue differences, and it is a positive control: the design and the '
    'analysis detect a real biological difference of this magnitude without difficulty '
    'in the same samples that show nothing for disease.')
print()
# The Venn in Figure 6A is the PAIRED filter split over these 290 entries. It is NOT the
# `set` column of tissue_mirnas.tsv, which is a full-cohort tissue-enrichment grouping
# built for M19 and counts 299 entries. Quoting the latter for the figure is the error
# the v2 Results pass had to correct (189/18/92 of 299 instead of 180/46/64 of 290), so
# both are printed here, labelled, rather than one standing in for the other.
per = MD.groupby('patient')['tissue_name'].nunique()
paired = set(per[per == 2].index)
pcols = [c for c in MX.columns if MD.loc[c, 'patient'] in paired]
psets = {t: set(de_filtered(MX[[c for c in pcols
                                if MD.loc[c, 'tissue_name'] == t]], MD, t)[0].index)
         for t in ('lens', 'tear')}
L, T = psets['lens'], psets['tear']
say(f'RESULTS -> Of the {len(L | T)} entries tested in the paired set, {len(L - T)} pass '
    f'the filter in lens only, {len(T - L)} in tear only and {len(L & T)} in both. This is '
    f'the split Figure 6A shows.')
print()
tm = tsv('stage_6', 'tissue_mirnas.tsv').rename(columns={'Unnamed: 0': 'entry'})
print('full-cohort tissue-enrichment sets (M19 input, 299 entries, NOT the Figure 6A Venn):')
print(tm.groupby('set').size().to_string())
print()
print('top tissue-enriched miRNAs by log2(lens/tear):')
print(tm.assign(entry=lambda d: d.entry.str.replace('hsa-', ''))
        .nlargest(5, 'lfc')[['entry', 'lens_cpm', 'tear_cpm', 'lfc', 'set']]
        .to_string(index=False, float_format=lambda v: f'{v:.1f}'))
print(tm.assign(entry=lambda d: d.entry.str.replace('hsa-', ''))
        .nsmallest(5, 'lfc')[['entry', 'lens_cpm', 'tear_cpm', 'lfc', 'set']]
        .to_string(index=False, float_format=lambda v: f'{v:.1f}'))
''')

md('''
### R10b. [APPENDIX] Functional enrichment of the tissue contrast, and why the disease contrast has none

No enrichment analysis of the disease contrast is possible: there is no gene set to
enrich. The tissue contrast was analysed instead, and the result is reported together
with the diagnostics that make it uninterpretable in its naive form.

![Tissue enrichment](../../plots/branches/BEA-dedup/stage_6/fig_tissue_enrichment.png)
![miRNA-level GSEA](../../plots/branches/BEA-dedup/stage_6/fig_mirna_gsea.png)
''')

code(r'''
head('R10b  enrichment, with its calibration')
te = tsv('stage_6', 'tissue_enrichment.tsv')
byt = te[te.q < .05].tissue.value_counts()
say(f'RESULTS -> Hypergeometric testing of {len(te)} GO biological processes against a '
    f'universe of every gene targeted by any miRNA we express gave '
    f'{int((te.q < .05).sum())} terms at BH q < 0.05 '
    f'({", ".join(f"{int(v)} in {k}" for k, v in byt.items()) or "none"}).')
oc = tsv('stage_6', 'mirna_ora_calibration.tsv')
print()
print(oc.to_string(index=False))
say('CAVEAT -> The annotation-depth-matched null is the reason this arm is reported as '
    'uninterpretable rather than as a finding. The observed term count swings by orders '
    'of magnitude with the top-N cutoff, and the random-set maximum reaches or exceeds '
    'the observed count in several rows - a matched null drawn from the same universe '
    'can produce what we observe. The naive version of this analysis, run against a '
    'generic gene background, returned between 0 and 2,193 significant terms depending '
    'only on the choice of denominator.')
for f_, lab in (('mirna_gsea.tsv', 'every mature miRNA counted separately'),
                ('mirna_gsea_seed.tsv', 'miRNAs collapsed to seed families')):
    ge = tsv('stage_6', f_)
    sg = ge[ge.q_gsea < .05]
    le = [m.strip() for row in sg.leading_edge.dropna() for m in str(row).split(',')]
    uniq = pd.Series(le).value_counts()
    top5 = uniq.head(5).sum() / max(len(le), 1)
    say(f'RESULTS -> miRNA-level GSEA on the lens-versus-tear ranking, {lab}: '
        f'{int(len(sg))} of {len(ge)} sets reach q < 0.05. Their leading edges are drawn '
        f'from {len(uniq)} distinct miRNAs, and the five most frequent supply '
        f'{100*top5:.0f}% of all leading-edge memberships.')
say('CAVEAT -> Collapsing seed-sharing miRNAs is the single choice that decides this '
    'analysis, and it changes the number of significant sets by an order of magnitude. '
    'Seed-sharing miRNAs have near-identical predicted targets, so leaving them '
    'uncollapsed counts one seed several times; the collapsed run is the defensible one '
    'and it is the one that finds least. Either way a handful of miRNAs supply most of '
    'the leading edges, so the output describes which tissue-specific miRNAs happen to '
    'be well annotated rather than pathway biology.')
say('CAVEAT -> The annotation itself is the limiting factor in lens. miR-184, the '
    'defining lens and cornea miRNA, carries 380 miRPathDB GO terms and not one is '
    'eye-related. Functional databases are built from cancer, immune and epithelial '
    'work; the lens-specific miRNA families here are largely unannotated, so an absence '
    'of lens enrichment is a statement about the databases and not about the tissue.')
''')

md('''
## R11. One candidate survives correction, and it should not be believed yet

miR-708-5p in tear fluid is the only miRNA in either tissue to survive multiple-testing
correction. Figure 3C shows every measurement behind it. The panel is in the main text
because it is the strongest thing in the dataset and it still does not look like much:
the distributions overlap substantially and the separation rests on a shift in a
low-abundance species.
''')

code(r'''
head('R11  miR-708-5p in tear')
E = 'hsa-miR-708-5p'
tear = TISSUE['tear']
grp = MD.loc[tear, 'disease']
raw = MX.loc[E, tear]
cpm = (MX[tear] / MX[tear].sum() * 1e6).loc[E]
row = de[(de.tissue == 'tear') & (de.entry == E)].iloc[0]
say(f'RESULTS -> miR-708-5p is lower in XFG tear: log2 fold change {row.ql_log2FC:+.2f} '
    f'({2**abs(row.ql_log2FC):.1f}-fold down), edgeR quasi-likelihood P = '
    f'{row.ql_p:.1e}, Benjamini-Hochberg q = {res["tear"]["minq"]:.3f}. It is the only '
    f'miRNA to survive correction in either tissue ({res["lens"]["n"]} tests in lens '
    f'and {res["tear"]["n"]} in tear, corrected within tissue).')
for g, lab in (('control', 'controls'), ('diseased', 'XFG')):
    r_, c_ = raw[grp == g], cpm[grp == g]
    say(f'RESULTS -> In {lab} (n = {int((grp == g).sum())}): median {r_.median():.1f} raw '
        f'reads ({c_.median():.0f} CPM), mean {r_.mean():.1f} '
        f'({c_.mean():.0f} CPM), {int((r_ == 0).sum())} sample(s) with zero reads.')
say(f'RESULTS -> At a median of {raw[grp == "control"].median():.1f} reads in controls '
    f'and {raw[grp == "diseased"].median():.1f} in XFG this is a low-abundance species, '
    f'and the group difference is a shift in a broad, overlapping distribution rather '
    f'than a separation.')

pm = pd.read_csv(P / 'stage_4' / 'perm_edger_tear.tsv.gz', sep='\t', index_col=0)
real, null = pm.iloc[0][E], pm.iloc[1:][E]
pf = (1 + (null <= real).sum()) / (1 + len(null))
best = pm.iloc[1:].min(axis=1)
fw = (1 + (best <= real).sum()) / (1 + len(best))
print()
say(f'RESULTS -> Across {len(null)} label permutations, miR-708-5p itself reached a '
    f'p-value this small in {100*pf:.1f}% of shuffles (per-feature permutation '
    f'p = {pf:.3f}).')
say(f'RESULTS -> But the miRNA was selected by screening 259 features. The question that '
    f'matches how it was found is how often a shuffle produces ANY feature this '
    f'extreme, and the answer is {100*fw:.0f}% (familywise permutation p = {fw:.2f}). '
    f'A single candidate of this strength is the expected yield of a screen this wide '
    f'on these data.')
stv = tsv('stage_4', 'stability.tsv')
srow = stv[(stv.tissue == 'tear') & (stv.entry == E)].iloc[0]
say(f'RESULTS -> It is robust to sample removal - worst leave-one-out P = '
    f'{srow.p_loo_worst:.3f}, and {100*srow.rand_frac_sig:.0f}% of random three-sample '
    f'deletions keep P < 0.05 - so it is not an artefact of one or two patients.')
say('RESULTS -> Taken together: the measurement is real and in the data, it does not '
    'depend on normalisation or on a handful of samples, and it is nevertheless what a '
    'screen of this width returns by chance about one time in eight. It is reported as a '
    'candidate requiring independent replication, not as a finding.')
print()
say('CAVEAT -> Two further reasons for caution, both of which belong in the Discussion. '
    'First, the tear miRNome reference identified miR-708-5p as one of two '
    'race-associated miRNAs in healthy tears (1.73-fold, adjusted P = 0.037) - it is '
    'among the species that vary most by demography in this compartment, which is '
    'precisely the behaviour that generates spurious disease associations in a '
    'case-control design. Our cohort is ancestry-homogeneous so this cannot confound us '
    'directly, but it bears on how the candidate should be interpreted.')
# Primary-arm intervals, same table panel C of the replication figures now writes.
# The old PEXlens-v22 lens_equivalence.tsv was retired 2026-10-21: it was pydeseq2 on
# median-of-ratios, with a permissive second fit this paper does not use anywhere.
le = tsv('replication', 'lens_claims_vs_our_CIs.tsv', branch='BEA-dedup').set_index('miRNA')
lr = le.loc[E]
ld = tsv('replication', 'lens_genes2022_de.tsv', branch='PEXlens-v22').set_index('entry')
say(f'CAVEAT -> Second, the direction does not match the one prior report of this miRNA '
    f'in exfoliation. The lens comparator reported miR-708-5p UP in exfoliation capsule '
    f'by {lr.theirs:+.2f} log2 units; reproducing their pipeline on our lens capsule '
    f'gives {ld.logFC.get(E, float("nan")):+.3f} (P = {ld.p.get(E, float("nan")):.2f}), '
    f'and our 90% interval [{lr.lo:+.2f}, {lr.hi:+.2f}] excludes their effect. Our own '
    f'signal is in tear and in the opposite direction. Shared nomenclature across '
    f'different tissues and opposite directions is a coincidence until something makes '
    f'it more than one.')
''')

md('''
## R12. [APPENDIX] The replication arms

Each comparator's pipeline, filter, test and threshold were applied to our data. The
purpose is the sample-similarity argument in R1; the differential-expression output of
these arms is reported for completeness and is where the filter dependence shows up.

![Lens composition](../../plots/branches/PEXlens-v22/replication/fig_lens_composition.png)
![Tear rank concordance](../../plots/branches/CBGM-v22/replication/fig_rank_concordance.png)
''')

code(r'''
head('R12  replication arms, differential expression')
ld = tsv('replication', 'lens_genes2022_de.tsv', branch='PEXlens-v22').set_index('entry')
THEIRS = ['hsa-miR-671-3p', 'hsa-miR-374a-5p', 'hsa-miR-1307-5p', 'hsa-miR-708-5p']
say(f'METHODS -> Under the lens comparator\'s own filter (3.7 CPM in >=50% of samples) '
    f'{len(ld)} miRNAs were testable in our 43 lens libraries, against the 226 that pass '
    f'our own filter - their filter is the more stringent of the two at our n.')
for e in THEIRS:
    if e in ld.index:
        r_ = ld.loc[e]
        say(f'RESULTS -> {e.replace("hsa-", "")}: log2FC {r_.logFC:+.3f}, P = {r_.p:.2f}, '
            f'FDR = {r_.FDR:.2f} - not reproduced.')
    else:
        say(f'RESULTS -> {e.replace("hsa-", "")} did not pass their own expression filter '
            f'in our data and could not be tested.')
say(f'RESULTS -> None of the four miRNAs they reported as differentially expressed is '
    f'reproduced at 4 to 9 times their sample size, under their pipeline and their test.')
say(f'RESULTS -> Their criteria applied to this arm return {int((ld.p < .1).sum())} '
    f'miRNAs at P < 0.1 and {int((ld.p < .05).sum())} at P < 0.05, from {len(ld)} '
    f'tested - comparable to the 6 they reported.')
print()
print(ld.nsmallest(5, 'p').assign(entry=lambda d: d.index.str.replace('hsa-', ''))[
        ['entry', 'logFC', 'p', 'FDR']].to_string(index=False,
                                                  float_format=lambda v: f'{v:.4g}'))
nz = ld[ld.FDR < .05]
if len(nz):
    say(f'CAVEAT -> One miRNA does survive FDR correction in this arm: '
        f'{", ".join(e.replace("hsa-", "") for e in nz.index)} '
        f'(FDR {nz.FDR.iloc[0]:.3f}). It is not one of theirs, and it does not survive '
        f'our own primary analysis of the same tissue '
        f'(raw P = {de[(de.tissue=="lens") & (de.entry==nz.index[0])].ql_p.iloc[0]:.4f}, '
        f'BH q = {res["lens"]["minq"]:.2f} at best, and it fails the leave-one-out test). '
        f'The difference is the expression filter: a more stringent filter tests fewer '
        f'features, so the same raw p-value carries a smaller correction. That '
        f'sensitivity to an upstream choice no one reports is itself worth stating.')
print()
sb = tsv('replication', 'tear_stability.tsv', branch='CBGM-v22').rename(
    columns={'Unnamed: 0': 'entry'})
ref = pd.read_csv(REFS / 'jones2026_tear_stable10.tsv', sep='\t')
col = [c for c in ref.columns if 'miR' in str(ref[c].iloc[0])][0]
ours10 = set(sb.nsmallest(10, 'total_stability').entry)
say(f'RESULTS -> Reproducing their stable-miRNA analysis, '
    f'{len(ours10 & set(ref[col]))} of their 10 most stable tear miRNAs appear in our '
    f'10 most stable.')
print(sb.nsmallest(10, 'total_stability')
        .assign(entry=lambda d: d.entry.str.replace('hsa-', ''))[
            ['entry', 'rank', 'CV', 'geNorm', 'NormFinder', 'total_stability']]
        .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
''')

md('''
## R13. [APPENDIX] Supplementary tables

The per-miRNA results are supplied in full rather than as a top-N list, so that any
reader can apply their own threshold. The cell below writes them.
''')

code(r'''
head('R13  supplementary tables')
SUP = ROOT / 'manuscript' / 'supplementary'
SUP.mkdir(parents=True, exist_ok=True)

eqf = tsv('stage_4', 'equivalence.tsv')
eqf = eqf[(eqf.method == 'edgeR-QL') & (eqf.norm == 'CPM')] if 'method' in eqf else eqf
full = []
for t in ('lens', 'tear'):
    s = de[de.tissue == t].set_index('entry')
    q = pd.Series(multipletests(s.ql_p.values, method='fdr_bh')[1], index=s.index)
    out = pd.DataFrame(dict(tissue=t, logCPM=s.logCPM.round(3),
                            log2FC=s.ql_log2FC.round(4), F=s.ql_F.round(3),
                            P=s.ql_p, BH_q=q.round(4)))
    e = eqf[eqf.tissue == t].set_index('entry') if 'entry' in eqf else pd.DataFrame()
    for c in ('se', 'ci_lo', 'ci_hi', 'achieved_bound', 'state_1', 'p_tost_1',
              'state_1p66'):
        if len(e) and c in e:
            out[c] = e[c].reindex(out.index)
    full.append(out.sort_values('P'))
full = pd.concat(full)
full.to_csv(SUP / 'table_S1_differential_expression.tsv', sep='\t')
print(f'S1  {len(full):4d} rows  every tested miRNA, both tissues, with 90% CI, '
      f'achieved bound and equivalence state  -> {SUP.name}/'
      f'table_S1_differential_expression.tsv')

for name, fr in (('table_S2_stability.tsv', tsv('stage_4', 'stability.tsv')),
                 ('table_S3_classifiers.tsv', cfg),
                 ('table_S4_tissue_contrast.tsv',
                  tsv('stage_6', 'tissue_disease_interaction.tsv')),
                 ('table_S5_published_criteria.tsv', cr),
                 ('table_S6_arm_comparison.tsv',
                  pd.read_csv(SHARED / 'arm_comparison.tsv', sep='\t')),
                 ('table_S7_excluded_entries.tsv',
                  pd.read_csv(SHARED / 'stage_1' / 'suspects_verified.tsv', sep='\t'))):
    fr.to_csv(SUP / name, sep='\t', index=False)
    print(f'{name.split("_")[1]:>3}  {len(fr):4d} rows  -> {SUP.name}/{name}')
print()
say('CAVEAT -> Table S2 names the library that drives each leave-one-out result. Those '
    'are pseudonymous library identifiers, already present in the deposited count '
    'matrices, and carry no clinical fields - but check that this is acceptable to the '
    'journal and the ethics approval before the table is submitted.')
say('METHODS -> Supplementary Table S1 lists every tested miRNA in both tissues with '
    'its effect size, p-value, BH q-value and equivalence state, so that no threshold '
    'used in this paper has to be taken on trust. Tables S2-S7 give the stability '
    'results, every classifier configuration, the tissue contrast, the published '
    'criteria applied here, the cross-pipeline comparison, and the read-level '
    'verification of excluded entries.')
''')

md('''
---
## Figure and table inventory

| item | script | file |
|---|---|---|
| Figure 1 | `scripts/manuscript/fig1.py` | `manuscript/figures/figure1.pdf` |
| Figure 2 | `fig2.py` | `figure2.pdf` |
| Figure 3 | `fig3.py` | `figure3.pdf` |
| Figure 4 | `fig4.py` | `figure4.pdf` |
| Figure 5 | `fig5.py` | `figure5.pdf` |
| Figure 6 | `fig6.py` | `figure6.pdf` |

**Numbering note.** The meeting plan had the classifier figure as Figure 4 and the Venn
as Figure 5. The published-criteria figure was added afterwards as Figure 4, pushing the
classifier to 5 and the tissue contrast to 6. The files and scripts follow the current
numbering; the plan does not.

### Appendix figures still to be drawn

* **covariate PCAs**: R2b computes every statistic; the panels themselves are not drawn.
  One small PCA per covariate, within each tissue, with the disease panel repeated for
  reference.
* **equivalence against Δ**: the full curve of excluded miRNAs as a function of the
  bound, promised in the Methods so that Δ = 1.0 cannot be read as cherry-picked.
  `fig_equivalence_field.png` shows the achieved-bound field, which is related but not
  the same figure.

Everything else referenced above is already drawn and is listed with its path at the
point of use.
''')


# build machinery

def notebook():
    cells = []
    for kind, src in CELLS:
        if kind == 'md':
            cells.append(dict(cell_type='markdown', metadata={}, id=f'md{len(cells)}',
                              source=src.splitlines(keepends=True)))
        else:
            cells.append(dict(cell_type='code', metadata={}, execution_count=None,
                              id=f'code{len(cells)}', outputs=[],
                              source=src.splitlines(keepends=True)))
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name='Python 3',
                                              language='python', name='python3'),
                              language_info=dict(name='python')))


def execute(nb):
    """Run the code cells in-process and attach their stdout."""
    ns = {}
    n_ok = 0
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c['source'])
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                exec(compile(src, f'<cell {i}>', 'exec'), ns)
        except Exception as e:
            import traceback
            c['outputs'] = [dict(output_type='stream', name='stdout',
                                 text=buf.getvalue().splitlines(keepends=True)),
                            dict(output_type='stream', name='stderr',
                                 text=traceback.format_exc().splitlines(keepends=True))]
            c['execution_count'] = i
            print(f'CELL {i} FAILED: {type(e).__name__}: {e}', file=sys.stderr)
            continue
        c['outputs'] = [dict(output_type='stream', name='stdout',
                             text=buf.getvalue().splitlines(keepends=True))]
        c['execution_count'] = i
        n_ok += 1
    return n_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exec', dest='run', action='store_true',
                    help='run the code cells and embed their output')
    a = ap.parse_args()
    nb = notebook()
    if a.run:
        import os
        os.chdir(ROOT)
        n = execute(nb)
        print(f'executed {n} code cells')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1) + '\n')
    nmd = sum(1 for k, _ in CELLS if k == 'md')
    print(f'wrote {OUT.relative_to(ROOT)}  '
          f'({nmd} markdown + {len(CELLS)-nmd} code cells)')


if __name__ == '__main__':
    main()
