#!/bin/bash
#SBATCH --job-name=w11-m1build
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile m1_gather.cu for sm_90 inside the CUDA devel image (no GPU needed to compile).
set -eu
WORK=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$WORK/images/cuda-12.4.1-devel-ubuntu22.04.sif
mkdir -p $WORK/jobs/w11/bin
apptainer exec $IMG nvcc --version
apptainer exec --bind $WORK/jobs/w11:$WORK/jobs/w11 $IMG nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo \
  -o $WORK/jobs/w11/bin/m1_gather $WORK/jobs/w11/m1_gather.cu
ls -l $WORK/jobs/w11/bin/m1_gather
