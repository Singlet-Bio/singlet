#!/bin/bash
#SBATCH --job-name=w11-ing-shuf
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: how much does the barcode sort of the source `.1fq` matter? Same transform with the pairs of every 500 K block in random order.
# Env: SAMPLE_LABEL, SAMPLE_1FQ. Output: JSON lines (sorted vs shuffled, zstd 1 and 3, 32K-pair chunks).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/ingest; M8=$W/jobs/w11/m8; BIN=$D/bin3
SAMPLE_LABEL=${SAMPLE_LABEL:?}; Q=${SAMPLE_1FQ:?}
OUT=$D/results/shuf_${SAMPLE_LABEL}_$SLURM_JOB_ID; mkdir -p $OUT; R=$OUT/results.jsonl
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
ctr() { apptainer exec -B $W --env LD_LIBRARY_PATH=$M8/lib $IMG "$@"; }
for L in 1 3; do
  for SH in 0 1; do
    ctr $BIN/enc_1fg --in $Q --out $S/x.1fg --codec zstd --level $L --threads 4 --crc 1 --chunk-pairs 32768 --shuffle $SH >> $R 2>> $OUT/err.txt
    sed -i "\$s/^{/{\"sample\":\"$SAMPLE_LABEL\",\"run\":\"zstd${L}_shuffle${SH}\",/" $R; rm -f $S/x.1fg
  done
done
cut -c1-120 $R
