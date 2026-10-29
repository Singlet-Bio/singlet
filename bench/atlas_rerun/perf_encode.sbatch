#!/bin/bash
#SBATCH --job-name=ar-encode
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch perf_encode.sbatch <dataset>
# Encode a benchmark dataset's FASTQs to .1fq once, so every perf config starts from the same file.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; DS=$1; D=$PB/data/$DS
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
/usr/bin/time -v -o $D/encode.time.txt $PB/bin/singlet-fixed encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz \
  -o $SCR/$DS.1fq > $D/encode.log 2>&1
cp $SCR/$DS.1fq $D/$DS.1fq; ls -la $D/$DS.1fq; grep -E "Elapsed|Maximum" $D/encode.time.txt
