# Reference tables

## Compiled for this study

These were built by reading the source papers and are our own work. Each pair of
`*.tsv` / `*_PROVENANCE.md` records what was checked and against which source.

| File | What it is |
|---|---|
| `glaucoma_mirna_studies.tsv` | The comparator literature: cohort sizes, control definition, assay, significance threshold, whether any multiple-testing correction was applied, and the number of hits reported. Backs Table 1 and the published-criteria analysis. |
| `glaucoma_mirna_studies_PROVENANCE.md` | How each field was verified, including corrections made after reading full texts. |
| `genes2022_lens_de.tsv` | The six differentially expressed miRNAs reported by Tomczyk-Socha et al. 2022, with log fold change and *P*. |
| `genes2022_lens_top10.tsv` | The ten most abundant lens-capsule miRNAs they report, reads and percent per group. |
| `jones2026_tear_top50.tsv`, `jones2026_tear_stable10.tsv` | The tear-fluid miRNome description reported by Jones et al. 2026. |
| `GSE291021_PROVENANCE.md` | How the Jones et al. count matrix was obtained from GEO and processed. |
| `references_verified.md` | Every citation in the manuscript, with the source each claim was checked against. |
| `tissue_marker_genes.tsv` | Lens and tear protein markers with citations, used by the tissue contrast. |

## Not redistributed

These files come from public databases and are not hosted here. Download each into
`refs/` under the name in the first column (rename where the source name differs); the
files are used unmodified.



| File | Source |
|---|---|
| `mature_v22.fa` | miRBase release 22, https://mirbase.org/download_version_files/22/mature.fa |
| `hairpin_v22.fa` | miRBase release 22, https://mirbase.org/download_version_files/22/hairpin.fa |
| `mature_current.fa` | miRBase release 23.1, https://mirbase.org/download_version_files/23.1/mature.fa |
| `GSE291021_jones2026_tear_counts.csv.gz` | GEO series GSE291021, file `GSE291021_TearFluidmiRNA_ProcessedData.csv.gz` from https://ftp.ncbi.nlm.nih.gov/geo/series/GSE291nnn/GSE291021/suppl/ (see `GSE291021_PROVENANCE.md`). Needed by `scripts/replication/fig_tear_jones_meansd.py` and `figs_genes2022.py`, and so by Figure 1. |
| `h.all.*.gmt`, `c2.*.gmt`, `c3.*.gmt`, `c5.*.gmt` | MSigDB, https://www.gsea-msigdb.org (registration required) |
| `miRDB_v6.0_prediction_result.txt.gz` | miRDB v6.0, https://mirdb.org |
| `miRPathDB_*.gmt` | miRPathDB v2.0, https://mpd.bioinf.uni-sb.de |
| `gtrnadb.fasta.gz`, `pirbase.fasta.gz`, `snodb.fasta.gz`, `snopy.fasta.gz`, `snorna_database.fasta.gz`, `refseq_human.fasta.gz`, `refGene.txt.gz` | The small-RNA class references used by the stage 1 read-level verification, from GtRNAdb, piRBase, snoDB, snOPY, snoRNA Database, RefSeq and UCSC respectively. |

