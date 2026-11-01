#!/bin/bash
#SBATCH --job-name=bp-ap-determ
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%A_%a.out
#   sbatch determinism.sbatch        then  sbatch determinism_compare.sbatch
# Which outputs change with the thread count? The pinned campaign binary on PBMC 1k v3 (.1fq), four runs, outputs kept:
#   0: 4 threads, cache on    1: 8 threads, cache on    2: 4 threads, SINGLET_STAR_SEQCACHE_MB=0    3: 8 threads, cache off
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
verify_bin
NAMES=(t4_cache t8_cache t4_nocache t8_nocache); THR=(4 8 4 8)
i=$SLURM_ARRAY_TASK_ID; N=${NAMES[$i]}; T=${THR[$i]}
REF=$REFBASE/GRCh38-2024-A; O=$BP/determinism/$N; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cp $PB/data/pbmc1k_v3/pbmc1k_v3.1fq $SCR/s.1fq
unset SINGLET_REF_BASE; export SINGLET_STALL_TIMEOUT=10800
case $N in *nocache) export SINGLET_STAR_SEQCACHE_MB=0 ;; esac
cd $SCR
$BIN $SCR/s.1fq --ref-base $REFBASE --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ \
  --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
rm -f $O/*.bam $O/*.bam.bai $O/star_Unmapped* $O/nonhost_reads.fq; du -sh $O
