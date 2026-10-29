#!/bin/bash
#SBATCH --job-name=w11-m6-compile
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# compile m6.cu for sm_90 in the CUDA devel image (no GPU needed to compile)
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; M=$W/jobs/w11/m6; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
mkdir -p $M/bin
for nw in ${NWS:-4}; do
  apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -DNW=$nw -lineinfo -o $M/bin/m6_nw$nw $M/m6.cu 2>&1 | grep -v "Wno-deprecated" | head -60
  ls -la $M/bin/m6_nw$nw
done
apptainer exec -B $W $IMG nvcc --version | tail -2
