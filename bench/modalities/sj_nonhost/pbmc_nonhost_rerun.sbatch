#!/bin/bash
#SBATCH --job-name=sjnh-pbmc-nh
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# PBMC 1k v3 (existing .1fq) with the non-host screen, nonhost/ pre-created so the per-cell species table is written.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; B=$W/modality-bench/sj_nonhost
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A; BIN=$PB/bin/singlet-fixed-det
O=$B/singlet/pbmc1k_v3; rm -rf $O; mkdir -p $O/nonhost
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap "rm -rf $SCR" EXIT; cp $PB/data/pbmc1k_v3/pbmc1k_v3.1fq $SCR/s.1fq
export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference SINGLET_STALL_TIMEOUT=10800
cd $SCR; /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $SINGLET_REF_BASE --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf \
  --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum" $O/time.txt; ls $O/nonhost
