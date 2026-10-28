#!/bin/bash
#SBATCH --job-name=w11-m5cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile bench/gpu/m5/m5.cu (with stitch.cuh) in the CUDA devel image -> $W/jobs/w11/bin/m5
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/m5
apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v \
  -o $W/jobs/w11/bin/m5.new $D/m5.cu 2>&1 | grep -v "^ptxas info    : Function properties\|bytes stack frame" | tail -25
mv $W/jobs/w11/bin/m5.new $W/jobs/w11/bin/m5 && echo "[compile] ok"
