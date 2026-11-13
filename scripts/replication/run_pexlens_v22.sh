#!/usr/bin/env bash
# PEXlens-v22: replication of the lens-capsule workflow from Genes 2022, 13(4), 582
#   (Tomczyk-Socha et al. - PEX miRNA expression by NGS).
#   https://pmc.ncbi.nlm.nih.gov/articles/PMC9031982/
# run as: scripts/replication/run_pexlens_v22.sh main [parallel]
#
# SPECIFIED BY THE PAPER: bowtie2 to miRBase mature miRNAs; miRBase March 2018
#   (= v22) -> WE NOW USE v22, matching them exactly. The original arm used v23 on
#   the grounds the release does not move counts; for a replication arm that is the
#   wrong trade (all four of their reported DE miRNAs are v22 names, and
#   cross-release comparison silently costs matches). Downstream TMM + edgeR LRT,
#   filter 3.7 CPM in >=5 samples, is the ANALYSIS layer (lens_genes2022.R); this
#   script produces the count matrix only.
#
# NOT SPECIFIED -> defaults / stated assumption:
#   - bowtie2 parameters: DEFAULTS (end-to-end; its scoring permits mismatches --
#     materially more permissive than the tear paper's bowtie1 -v 0)
#   - adapter trimming: not described in the paper; same cutadapt step as the
#     CBGM-like arm so the arms differ ONLY in aligner. NO length filter (they
#     state none).
#   - counting: primary alignments only (SAM flag 0x904 excluded); no UMI handling
#     (their Lexogen kit has none)
set -euo pipefail
COHORT=${1:?usage: run_pexlens_like.sh pilot_OR_main [parallel]}
PAR=${2:-6}
# Parent of Glaugene2/ and of the NGI delivery P40752/. Override for another
# machine: GG2_STUDY_ROOT=/path/to/study ./<script>
ROOT=${GG2_STUDY_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
BASE=$ROOT/Glaugene2/counts/PEXlens-v22
REF=$BASE/ref
OUT=$BASE/$COHORT
CUTADAPT=quay.io/biocontainers/cutadapt:4.0--py39hbf8eff0_0
BT2=quay.io/biocontainers/bowtie2:2.5.4--he96a11b_5
QIA=AACTGTAGGCACCATCAAT
mkdir -p "$OUT"

one () {
  S=$1; FQ=$(readlink -f "$2"); D=$(dirname "$FQ")   # resolve symlinks: P40752/*/02-FASTQ points into fastq/main
  [ -s "$OUT/$S.counts.tsv" ] && { echo "$S: cached"; return 0; }
  # same trimming as the CBGM-like arm, but no 21-24nt window
  docker run --rm -v "$OUT:/w" -v "$D:/fq:ro" -w /w -u 1000:1000 $CUTADAPT \
    cutadapt -a $QIA -m 16 -j 4 -o ${S}.trim.fq.gz "/fq/$(basename "$FQ")" \
    > $OUT/${S}.cutadapt.log 2>&1
  docker run --rm -v "$OUT:/w" -v "$REF:/ref:ro" -w /w -u 1000:1000 $BT2 \
    sh -c "bowtie2 -p 4 -x /ref/mature_v22 -U ${S}.trim.fq.gz -S ${S}.sam" \
    > $OUT/${S}.bowtie2.log 2>&1
  awk '!/^@/ && and($2,0x904)==0 {c[$3]++} END{for(k in c) print k"\t"c[k]}' \
      "$OUT/${S}.sam" | grep -v '^\*' | sort > "$OUT/${S}.counts.tmp"
  mv "$OUT/${S}.counts.tmp" "$OUT/${S}.counts.tsv"
  rm -f "$OUT/${S}.sam" "$OUT/${S}.trim.fq.gz"
  echo "$S: $(wc -l < "$OUT/${S}.counts.tsv") miRNAs, $(awk -F'\t' '{s+=$2}END{print s+0}' "$OUT/${S}.counts.tsv") reads"
}
export -f one; export OUT REF CUTADAPT BT2 QIA

if [ "$COHORT" = pilot ]; then
  while read -r B S; do
    echo "$S $(ls $ROOT/BEA/documents_20260819/BEA25P077_RP_${B}_S*_R1_001.fastq.gz)"
  done <<< "1 s1A_G2
2 s2A_G2
3 s1B_G2
4 s2B_G2
5 s3A_G2
6 s4A_G2
7 s5A_G2
8 s6A_G2
21 s15B_G2
22 s16B_G2
23 s17B_G2
24 s18B_G2"
else
  cut -f1 $ROOT/P40752/sample_metadata.tsv | tail -n +2 | while read -r S; do
    echo "$S $(find $ROOT/P40752/$S -name '*R1_001.fastq.gz' | head -1)"
  done
fi | xargs -P "$PAR" -L1 bash -c 'one "$0" "$1"'
echo "ALLDONE $COHORT"
