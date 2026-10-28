#!/bin/bash
#SBATCH --job-name=w11-m5cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile bench/gpu/m5/m5.cu (with stitch.cuh) in the CUDA devel image -> $W/jobs/w11/bin/m5 and lane-A capacity variants
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/m5
cc() { apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v "$@" 2>&1 \
       | grep -E "error|k_lane|registers" | tail -8; }
cc -o $W/jobs/w11/bin/m5.new $D/m5.cu &
cc -DLA_NW=64 -DLA_NWA=8 -DLA_NWB=1024 -o $W/jobs/w11/bin/m5w.new $D/m5.cu &
cc -DLA_NW=32 -DLA_NWA=8 -DLA_NWB=512 -o $W/jobs/w11/bin/m5m.new $D/m5.cu &
wait
for b in m5 m5w m5m; do mv $W/jobs/w11/bin/$b.new $W/jobs/w11/bin/$b; done && echo "[compile] ok"
