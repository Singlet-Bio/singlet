#!/bin/bash
#SBATCH --job-name=w11-m1knee
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# M1 follow-up: locate the footprint knee between 64 and 80 GB (malloc only).
set -u
WORK=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$WORK/images/cuda-12.4.1-devel-ubuntu22.04.sif
OUT=$WORK/jobs/w11/results; mkdir -p $OUT
echo "node=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi; nvidia-smi -q -d CLOCK; nvidia-smi -q | grep -E "Driver Version|CUDA Version|Product Name|Persistence|Compute Mode"
apptainer exec --nv $IMG nvcc --version
apptainer exec --nv -B $WORK $IMG $WORK/jobs/w11/bin/m1_gather --gb 66,68,70,72,74,76,78 --reps 3 --alloc malloc > $OUT/m1_knee.jsonl
echo "exit=$?"; wc -l $OUT/m1_knee.jsonl
nvidia-smi -q -d CLOCK | grep -A4 -E "^    Clocks$|Max Clocks"
