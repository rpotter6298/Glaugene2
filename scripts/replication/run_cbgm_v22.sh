#!/usr/bin/env bash
# CBGM-v22: the CBGM-like pipeline re-run against miRBase v22, matching the
# comparator exactly (Jones et al. 2026, Ocul Surf 40:133-143). The original
# CBGM-like arm used v23 on the grounds the release does not move counts; for a
# replication arm that is the wrong trade - the comparator used v22.1 and
# cross-release comparison silently costs matches (v22 `miR-184` is v23
# `miR-184-3p` and so on; three of six apparent misses in their top 50 were naming,
# not biology). Running on v22 removes the mapping step entirely.
# Everything else is identical to scripts/run_cbgm_like.sh (one level up, outside
# this repo): Cutadapt 4.0, reads 21-24 nt, bowtie1 1.3.1 -v 0, primary alignments
# only, mature miRNAs only, no UMI dedup.
# usage: scripts/replication/run_cbgm_v22.sh [parallel]
set -euo pipefail
PAR=${1:-6}
# Parent of Glaugene2/ and of the NGI delivery P40752/. Override for another
# machine: GG2_STUDY_ROOT=/path/to/study ./<script>
ROOT=${GG2_STUDY_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
BASE=$ROOT/Glaugene2/counts/CBGM-v22
REF=$BASE/ref
OUT=$BASE/main
CUTADAPT=quay.io/biocontainers/cutadapt:4.0--py39hbf8eff0_0
BOWTIE=quay.io/biocontainers/bowtie:1.3.1--py310h4070885_4
QIA=AACTGTAGGCACCATCAAT
mkdir -p "$OUT"

one () {
  S=$1; FQ=$(readlink -f "$2"); D=$(dirname "$FQ")   # resolve: P40752/*/02-FASTQ holds symlinks into fastq/main with different names
  [ -s "$OUT/$S.counts.tsv" ] && { echo "$S: cached"; return 0; }
  docker run --rm -v "$OUT:/w" -v "$D:/fq:ro" -w /w -u 1000:1000 $CUTADAPT \
    cutadapt -a $QIA -m 21 -M 24 -j 4 -o ${S}.trim.fq.gz "/fq/$(basename "$FQ")" \
    > $OUT/${S}.cutadapt.log 2>&1
  docker run --rm -v "$OUT:/w" -v "$REF:/ref:ro" -w /w -u 1000:1000 $BOWTIE \
    sh -c "bowtie -v 0 -k 50 --best --strata -p 4 --sam -x /ref/mature_v22 \
           -q ${S}.trim.fq.gz ${S}.sam" > $OUT/${S}.bowtie.log 2>&1
  awk '!/^@/ && and($2,0x904)==0 {c[$3]++} END{for(k in c) print k"\t"c[k]}' \
      "$OUT/${S}.sam" | sort > "$OUT/${S}.counts.tmp"
  mv "$OUT/${S}.counts.tmp" "$OUT/${S}.counts.tsv"
  rm -f "$OUT/${S}.sam" "$OUT/${S}.trim.fq.gz"
  echo "$S: $(wc -l < "$OUT/${S}.counts.tsv") miRNAs, $(awk -F'\t' '{s+=$2}END{print s}' "$OUT/${S}.counts.tsv") reads"
}
export -f one; export OUT REF CUTADAPT BOWTIE QIA

cut -f1 $ROOT/P40752/sample_metadata.tsv | tail -n +2 | while read -r S; do
  echo "$S $(find $ROOT/P40752/$S -name '*R1_001.fastq.gz' | head -1)"
done | xargs -P "$PAR" -L1 bash -c 'one "$0" "$1"'
echo "ALLDONE CBGM-v22"
