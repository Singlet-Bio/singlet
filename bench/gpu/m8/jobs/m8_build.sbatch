#!/bin/bash
#SBATCH --job-name=w11-m8build
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build m8_ingest (sm_90) in the CUDA devel image; zstd from the conda env pbench, .1fq headers from the uploaded include/ tree.
set -eu
WORK=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$WORK/images/cuda-12.4.1-devel-ubuntu22.04.sif
PB=$WORK/envs/pbench
M8=$WORK/jobs/w11/m8
mkdir -p $M8/bin
ls -l $PB/lib/libzstd.a 2>&1 || true
LIBZ="-L$PB/lib -lzstd"; [ -f $PB/lib/libzstd.a ] && LIBZ=$PB/lib/libzstd.a
apptainer exec -B $WORK $IMG nvcc --version
apptainer exec -B $WORK $IMG nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo -I$M8/include -I$PB/include \
  -Xcompiler -Wall,-Wno-unused-function,-pthread -o $M8/bin/m8_ingest $M8/m8_ingest.cu $LIBZ -lpthread
ls -l $M8/bin/m8_ingest
ldd $M8/bin/m8_ingest | head
