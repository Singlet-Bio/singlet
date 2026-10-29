#!/bin/bash
#SBATCH --job-name=w11-m34cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile bench/gpu/m3m4/seedsearch.cu (uploaded with its header) in the CUDA devel image -> $W/jobs/w11/bin/seedsearch
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/m3m4   # contains bench/gpu/m3m4/seedsearch.cu and src/gpu/stardump/dump_format.h in repo layout
apptainer exec -B $W $IMG nvcc --version | tail -2
apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo -o $W/jobs/w11/bin/seedsearch.new $D/bench/gpu/m3m4/seedsearch.cu
mv $W/jobs/w11/bin/seedsearch.new $W/jobs/w11/bin/seedsearch && echo "[compile] ok"
