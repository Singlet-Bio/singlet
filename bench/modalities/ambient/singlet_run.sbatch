#!/bin/bash
#SBATCH --job-name=amb-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch singlet_run.sbatch
# Singlet (singlet-fixed-det) on 10x hgmm_1k_v3 (HEK293T + NIH3T3) with the joint GRCh38+GRCm39 reference,
# keeping the raw barcode x gene matrix so the other tools can be run on the same counts.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; D=$PB/data/hgmm1k_v3
REFBASE=/mnt/projects/debruinz_project/cellarium/reference; REF=$REFBASE/GRCh38-GRCm39-barnyard
BIN=$PB/bin/singlet-fixed-det; O=$W/modality-bench/ambient/singlet; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1
rm -f $SCR/R?.fastq.gz
export SINGLET_STALL_TIMEOUT=7200
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REFBASE --genome-dir $REF/star_2.7.11b \
  --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix \
  > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
$BIN --help > $O/help.txt 2>&1 || true
grep -E "Elapsed|Maximum" $O/time.txt; ls $O
