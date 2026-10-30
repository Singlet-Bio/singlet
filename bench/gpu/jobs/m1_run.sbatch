#!/bin/bash
#SBATCH --job-name=w11-m1
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# M1: random-gather / pointer-chase roofline on one H100 NVL. Record versions and clocks, then run the sweep.
set -u
WORK=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$WORK/images/cuda-12.4.1-devel-ubuntu22.04.sif
OUT=$WORK/jobs/w11/results; mkdir -p $OUT
echo "node=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi; nvidia-smi -q -d CLOCK; nvidia-smi -q | grep -E "Driver Version|CUDA Version|Product Name|Persistence|Compute Mode"
apptainer exec --nv $IMG nvcc --version
apptainer exec --nv -B $WORK $IMG $WORK/jobs/w11/bin/m1_gather --reps 3 --alloc both > $OUT/m1_raw.jsonl
echo "exit=$?"; wc -l $OUT/m1_raw.jsonl
nvidia-smi -q -d CLOCK | grep -A4 -E "^    Clocks$|Max Clocks"
