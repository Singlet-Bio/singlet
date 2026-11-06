#!/bin/bash
#SBATCH --job-name=gd-readlen
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# P3-ENG-15/16: share of corpus and atlas-like samples with cDNA reads over 128 bases (bench/gpu_device/read_lengths.py, SRA metadata only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; OUT=$W/jobs/gpu-device/results/readlen-$SLURM_JOB_ID; mkdir -p $OUT
python3 -I $W/jobs/gpu-device/read_lengths.py $W/benchmark-store/datasets.tsv $OUT ${N_ATLAS:-3000} ${SEED:-1}
echo "results: $OUT"
