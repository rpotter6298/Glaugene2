# Count matrices

Nine quantifications of the same 95 main-cohort libraries. Each file is a
miRNA-by-sample matrix of raw counts, rows keyed by miRBase mature name and
columns by sequencing sample ID. `scripts/common/gg2.py` maps each arm name to
its file and to the miRBase release its names belong to.

| arm | file | miRBase | method |
|---|---|---|---|
| `BEA-raw` | `BEA-raw/counts_BEA-raw_main.tsv` | v22 | the facility's delivered counts, not deduplicated (STAR 2.7.11b, featureCounts 2.0.8) |
| `BEA-dedup` | `BEA-dedup/counts_BEA-dedup_main.tsv` | v22 | the facility's delivered counts after UMICollapse 1.0 |
| `BEA-like` | `BEA-like/counts_BEA_main.tsv` | v22 | our reproduction of that method from the FASTQs (fastp, subread-align local) |
| `nfcore` | `nfcore/counts_nfcore_main.tsv` | v23 | nf-core/smrnaseq: fastp, bowtie1 `-n 2` against mature and hairpin, mirtop |
| `nfcore-dedup` | `nfcore-dedup/counts_nfcore-dedup_main.tsv` | v23 | the same, counting unique (insert, 12 nt UMI) molecules instead of reads |
| `CBGM-like` | `CBGM-like/counts_CBGM_main.tsv` | v23 | Cutadapt 4.0, bowtie1 1.3.1 `-v 0`, reads 21-24 nt, mature only |
| `CBGM-v22` | `CBGM-v22/counts_CBGM-v22_main.tsv` | v22 | the same against v22, to match the comparator's release exactly |
| `PEXlens-like` | `PEXlens-like/counts_PEXlens_main.tsv` | v23 | Cutadapt 4.0 with a 16 nt minimum, bowtie2 2.5.4 end-to-end defaults, mature only |
| `PEXlens-v22` | `PEXlens-v22/counts_PEXlens-v22_main.tsv` | v22 | the same against v22 |

`BEA-dedup` is the primary arm, and every script follows `gg2.PRIMARY` rather
than naming an arm. Set `GG2_ARM=<name>` to run the pipeline on another.

`nfcore-dedup/counts_nfcore-dedup-exact_main.tsv` is a second deduplication of
the same libraries, on exact (insert, UMI) pairs rather than the directional
rule, kept for the comparison in `scripts/replication/`.

`BEA-dedup/reproduction/` holds seven samples quantified under both candidate
deduplication rules alongside the delivered raw and deduplicated counts, which
is the input to `scripts/replication/fig_bea_dedup_reproduction.py`.

Three BEA arms use miRBase v22 and the rest v23. Any comparison across that
boundary must map names by sequence identity first, or `hsa-miR-184` and
`hsa-miR-184-3p` will silently fail to join; `gg2.name_map_v22_v23()` does this
and the scripts all use it.

