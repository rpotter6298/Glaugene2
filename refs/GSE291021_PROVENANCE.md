# GSE291021 — Jones et al. 2026 tear fluid miRNA counts

Downloaded 2026-09-09 from
ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE291nnn/GSE291021/suppl/GSE291021_TearFluidmiRNA_ProcessedData.csv.gz

Series : GSE291021, "A Comprehensive Profile of microRNAs in Human Tear Fluid"
Paper  : Jones et al., Ocul Surf 2026;40:133-143. doi:10.1016/j.jtos.2026.01.011
Content: raw miRNA counts, 1,278 features x 32 healthy adult tear samples
Kit    : QIAseq miRNA Library Kit — the SAME kit we used, which is what makes this
         dataset able to separate kit effects from cohort effects.
Their pipeline: Cutadapt 4.0, 21-24 nt filter, bowtie1 1.3.1 -v 0, miRBase v22.1,
         SAMtools counts. Reproduced by our CBGM-v22 arm.

Not hosted in this repository. Download the file above and save it unmodified as
refs/GSE291021_jones2026_tear_counts.csv.gz. Cite the paper, not this repo.
