# GlauGene2 — analysis code

Code for *MicroRNA profiling reveals no clear differences between exfoliative glaucoma
patients and controls in lens capsule and tear fluid*.

Small RNA sequencing of anterior lens capsule (n = 43) and tear fluid (n = 51) from 52
patients undergoing cataract surgery, comparing exfoliative glaucoma (XFG) against
cataract-only controls. The study reports a negative result, so most of this code exists
to establish how far the absence of a signal can be trusted: equivalence testing,
permutation calibration, resampling at the sample sizes used in the comparator
literature, and the same analysis repeated across four quantification pipelines.

## Patient Data

This repository contains only the minimal metadata needed to run the primary analysis. 

## Layout

```
scripts/
  common/        gg2.py: sample metadata, count matrices, miRBase name mapping,
                 artefact and sample exclusions. Everything else imports this.
  stage_1/       read-level verification: which miRBase entries are really present
                 in the reads, and which are artefacts of a near-match elsewhere
  stage_2/       PCA and hierarchical clustering
  stage_3/       differential expression (edgeR-QL, NBSR, limma-voom, Box-Cox)
                 across four normalisations
  stage_4/       correction, equivalence testing, permutation calibration,
                 stability, small-study resampling
  stage_5/       random forest classification and its permutation null
  stage_6/       paired lens-versus-tear contrast, and the tissue x disease
                 interaction model
  replication/   reproductions of the comparator studies' pipelines, and of the
                 sequencing facility's own quantification
  report/        arm_comparison.py, which builds Supplementary Table S5
  manuscript/    the figure scripts, mstyle.py (shared journal style),
                 build_table_criteria.py and build_table_excluded.py
                 (Supplementary Tables S4 and S6), and build_notebook.py,
                 which regenerates the Methods and Results numbers from the
                 stored outputs
counts/          the nine count matrices, one per quantification arm, raw counts
                 by miRNA and sample (see counts/README.md)
refs/            reference tables compiled for this study (see refs/README.md)
manuscript/
  figures/       Figures 1-5 and Supplementary Figures S1-S4, PDF and 400 dpi PNG
  supplementary/ Supplementary Tables S1-S7, tab-delimited
```

Scripts are run directly and write into `plots/`, which is not tracked here. Most take
the quantification arm from the `GG2_ARM` environment variable and default to
`BEA-dedup`, the primary arm:

```
GG2_ARM=BEA-dedup python3 scripts/stage_3/de_edger.py
```

## Requirements

Python 3.11 or later with numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib,
openpyxl and joblib. The edgeR arms shell out to R with edgeR 4.x; `nbsr` is needed only
for the NBSR arm. The replication pipelines run cutadapt, bowtie1, bowtie2 and
UMICollapse through Docker, pinned to the biocontainers images named in each script.


**Sequencing data.** The assembled count matrices are in `counts/`. The reads are deposited
separately; see the data availability statement in the paper.

**Public reference files.** Files we obtained from public databases (miRBase, GEO, MSigDB
and others) are not hosted here. `refs/README.md` lists each one with where to download it
and the file name the scripts expect. The two miRBase `mature` files are needed by nearly
every analysis; the others only by the stages that use them.


## Reference tables in `refs/`

The TSVs here were compiled by reading the comparator papers, and each carries its own
provenance file recording what was verified and from which source. 
