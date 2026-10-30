#!/bin/bash
#SBATCH --job-name=bs-xs
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<id> phase4_extsort.sbatch      (one exclusive node, ~1 node-h)
# Phase 4b: external sort on the node's NVMe /tmp. STAR's sorted BAM is kept (SINGLET_KEEP_STAR_BAM=1), hash-shuffled with
# `samtools collate` to stand in for STAR's unsorted output (thread order), then sorted with `samtools sort` at 4 and 16 threads.
# Compare with the STAR in-run sort of the same sample (phase 1). Timings only; nothing is kept.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/singlet-bs-dev
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET
O=$W/bam-stream/p4/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bsxs-$SLURM_JOB_ID; mkdir -p $SCR/out $SCR/t; trap 'rm -rf $SCR' EXIT
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF SINGLET_KEEP_STAR_BAM=1
{ echo "node=$(hostname) $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"; sha256sum $BIN; df -h /tmp | tail -1; } | tee $O/node.txt
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
cd $SCR
$BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 4 \
  --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off "${SCREEN[@]}" > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
BAM=$(ls $SCR/out/*Aligned.sortedByCoord.out.bam); ls -l $BAM | tee $O/bam.txt
t() { local n=$1; shift; local s=$(date +%s.%N); "$@"; local e=$(date +%s.%N); echo "$n $(echo "$e - $s" | bc)" | tee -a $O/times.txt; }
t collate samtools collate -@ 16 -u -o $SCR/unsorted.bam $BAM $SCR/t/c
ls -l $SCR/unsorted.bam | tee -a $O/bam.txt
for th in 4 16; do
  t sort_t$th samtools sort -@ $th -m 3G -l 0 -T $SCR/t/s$th -o $SCR/sorted_t$th.bam $SCR/unsorted.bam
  t index_t$th samtools index -@ $th $SCR/sorted_t$th.bam
done
cmp <(samtools view $BAM | md5sum) <(samtools view $SCR/sorted_t16.bam | md5sum) && echo "record order identical to STAR sorted BAM: yes" | tee -a $O/times.txt || echo "record order identical to STAR sorted BAM: no (ties in coordinate order)" | tee -a $O/times.txt
grep -E "STAR completed|BAM indexed|refresh" $O/run.log | tee -a $O/times.txt
