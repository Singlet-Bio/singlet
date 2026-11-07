#!/bin/bash
#SBATCH --job-name=ir-route
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# input-routes: one new input route on one DEVELOPMENT library (never a held-out corpus row: sprint/intel/corpus-engine-touch.md in
# singlet-agents lists them). ROUTE picks the case; BIN is singlet-gpu of build.sbatch. Each case stages the reads with `singlet download`,
# records the protocol, read count and barcode-read geometry from the log and the .1fq header, and (PIPE=1, default) runs the CPU engine
# on the staged .1fq and keeps summary.json and the cell count. Output: $W/jobs/input-routes/routes/<ROUTE>-<job>/.
#   bam_pbmc1k     Cell Ranger 3 BAM of 10x pbmc_1k_v3 (a gate library) against the FASTQ folder route of the same library: read multisets
#   bam_kang_stim  Cell Ranger 1.x BAM of Kang 2018 batch 2 stimulated lane (GSE96583, not a corpus row), 10x 3' v1
#   bam_kang_head  the first HEADGB GB (default 3) of the same BAM by HTTP range (ENA serves it slowly), staged with --bam on the local prefix
#   gemcode_pbmc3k 10x GemCode FASTQ tar of pbmc3k (Cell Ranger 1.1, 10x 3' v1; not a corpus row), from the URL
#   dropseq_r1_76  Mereu 2020 Drop-seq library N703 (SRR9621743; the corpus row is N705), R1 76 nt, first MAXR spots
#   seqwell_r1_32  Gierahn 2017 Seq-Well HEK single cells (SRR5250843; the corpus row is SRR5250839), R1 32 nt, first MAXR spots
#   spec_<tag>     SRR staged with --protocol <tag> (first MAXR spots), for the protocol specs added on this branch
#   auto           SRR staged with auto-detection (first MAXR spots), to see what the engine calls a development library on its own
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?BIN: singlet-gpu}; ROUTE=${ROUTE:?ROUTE}; T=${SLURM_CPUS_PER_TASK:-16}; MAXR=${MAXR:-4000000}; PIPE=${PIPE:-1}
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF
O=$W/jobs/input-routes/routes/$ROUTE-$SLURM_JOB_ID; mkdir -p $O
SCR=/tmp/$USER-ir-rt-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) bin=$BIN commit=$(cat $(dirname $BIN)/commit-gpu 2>/dev/null) route=$ROUTE $(date -Is)" | tee $O/summary.txt
G=$REF/GRCh38-2024-A
stage() {   # name, download args...
  local n=$1; shift
  /usr/bin/time -v -o $O/$n.time.txt $BIN download "$@" -o $SCR/$n.1fq --staging-dir $SCR/stage-$n --vdb-threads 8 > $O/$n.log 2>&1
  local rc=$?
  echo "$n: exit $rc wall $(awk -F': ' '/Elapsed/{print $2}' $O/$n.time.txt) size $(stat -c %s $SCR/$n.1fq 2>/dev/null)" | tee -a $O/summary.txt
  grep -aE "Protocol:|Done:|FASTQ layout|input spec|swap|R1 trimmed|R1 .*→|barcodes:|ABORT|ERROR|WARNING" $O/$n.log | cut -c1-300 | sed "s/^/  $n | /" | tee -a $O/summary.txt
  $BIN decode $SCR/$n.1fq --header > $O/$n.header.txt 2>&1; sed "s/^/  $n header | /" $O/$n.header.txt | head -30 | tee -a $O/summary.txt
  return $rc
}
pipe() {   # name, genome dir
  [ "$PIPE" = 1 ] || return 0
  local n=$1 g=${2:-$G}; mkdir -p $SCR/run-$n/out
  ( cd $SCR/run-$n && /usr/bin/time -v -o $O/$n.pipe.time.txt $BIN $SCR/$n.1fq --ref-base $REF --genome-dir $g/star_2.7.11b --exons $g/genes/genes.gtf \
      --out-prefix $SCR/run-$n/out/ --threads $T --pipeline --output-format 1pz > $O/$n.pipe.log 2>&1 )
  echo "$n pipeline: exit $(awk -F': ' '/Exit status/{print $2}' $O/$n.pipe.time.txt) wall $(awk -F': ' '/Elapsed/{print $2}' $O/$n.pipe.time.txt)" | tee -a $O/summary.txt
  cp $SCR/run-$n/out/summary.json $O/$n.summary.json 2>/dev/null
  python3 - $O/$n.summary.json <<'EOF' 2>/dev/null | sed "s/^/  $n | /" | tee -a $O/summary.txt
import json, sys
d = json.load(open(sys.argv[1]))
def walk(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from walk(v, p + "." + k if p else k)
    else: yield p, x
for k, v in walk(d):
    if any(s in k.lower() for s in ("n_cells", "cells_called", "num_cells", "median_genes", "median_umi", "mapping_rate", "reads_total", "fraction_reads_in_cells", "protocol")):
        print(k, v)
EOF
}
readset() {   # name -> sha of the sorted (R1, R2) sequence pairs, and their count
  $BIN decode $SCR/$1.1fq --r1 $SCR/$1.r1.fq --r2 $SCR/$1.r2.fq --threads 8 > /dev/null 2>&1
  paste <(awk 'NR%4==2' $SCR/$1.r1.fq) <(awk 'NR%4==2' $SCR/$1.r2.fq) | LC_ALL=C sort -S 24G --parallel=$T -T $SCR > $SCR/$1.pairs
  rm -f $SCR/$1.r1.fq $SCR/$1.r2.fq
  echo "$1 read pairs $(wc -l < $SCR/$1.pairs) sha $(sha256sum $SCR/$1.pairs | cut -c1-16)" | tee -a $O/summary.txt
}
case $ROUTE in
  bam_pbmc1k)
    stage bam --bam https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_possorted_genome_bam.bam
    stage fqdir $PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs
    readset bam; readset fqdir
    echo "pairs only in bam $(LC_ALL=C comm -23 $SCR/bam.pairs $SCR/fqdir.pairs | wc -l), only in fastq $(LC_ALL=C comm -13 $SCR/bam.pairs $SCR/fqdir.pairs | wc -l)" | tee -a $O/summary.txt
    LC_ALL=C comm -3 $SCR/bam.pairs $SCR/fqdir.pairs | head -20 > $O/pairs_diff_head.txt
    rm -f $SCR/*.pairs
    pipe bam; pipe fqdir ;;
  bam_kang_stim)
    stage kang --bam https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398239/bam/2.2.bam ${EXTRA_ARGS:-}
    pipe kang ;;
  bam_kang_head)
    mkdir -p $SCR/kh; curl -fsS --retry 5 --retry-all-errors -r 0-$(( ${HEADGB:-3} * 1000000000 )) -o $SCR/kh/2.2.head.bam https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398239/bam/2.2.bam
    echo "prefix bytes $(stat -c %s $SCR/kh/2.2.head.bam)" | tee -a $O/summary.txt
    stage kang --bam $SCR/kh/2.2.head.bam ${EXTRA_ARGS:-}
    pipe kang ;;
  gemcode_pbmc3k)
    stage pbmc3k https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_fastqs.tar ${EXTRA_ARGS:-}
    pipe pbmc3k ;;
  dropseq_r1_76)
    stage auto SRR9621743 --max-reads $MAXR --no-profile || true
    stage dropseq SRR9621743 --max-reads $MAXR --protocol dropseq ${EXTRA_ARGS:-}
    pipe dropseq ;;
  seqwell_r1_32)
    stage auto SRR5250843 --max-reads $MAXR --no-profile || true
    stage seqwell SRR5250843 --max-reads $MAXR --protocol seqwell ${EXTRA_ARGS:-}
    pipe seqwell ;;
  spec_*)
    tag=${ROUTE#spec_}
    stage $tag ${SRR:?SRR} --max-reads $MAXR --protocol $tag ${EXTRA_ARGS:-}
    pipe $tag ${GENOME:-$G} ;;
  auto)
    stage auto ${SRR:?SRR} --max-reads $MAXR ${EXTRA_ARGS:-}
    pipe auto ${GENOME:-$G} ;;
  *) echo "unknown ROUTE $ROUTE"; exit 2 ;;
esac
echo "done $(date -Is)" | tee -a $O/summary.txt
