#!/bin/bash
#SBATCH --job-name=w11-ing-build
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build enc_1fg (host, g++) and read_1fg (nvcc, sm_90, nvCOMP 6) in the CUDA devel image. CPU only (no GPU gres).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
PB=$W/envs/pbench
M8=$W/jobs/w11/m8
D=$W/jobs/w11/ingest; ND=$D/nvcomp
SRC=${SRC:-$D/src}
BIN=${BIN:-$D/bin}; mkdir -p $BIN; ln -sf libnvcomp.so.6 $ND/lib64/libnvcomp.so
LIBZ="-L$PB/lib -lzstd"; [ -f $PB/lib/libzstd.a ] && LIBZ=$PB/lib/libzstd.a
ls $M8/include/singlet/fq | head -3
apptainer exec -B $W $IMG g++ -O3 -std=c++17 -march=x86-64-v3 -msse4.2 -Wall -Wno-unused-function -I$M8/include -I$PB/include -I$SRC -pthread \
  -o $BIN/enc_1fg $SRC/enc_1fg.cpp $LIBZ
apptainer exec -B $W $IMG g++ -O3 -std=c++17 -march=x86-64-v3 -msse4.2 -Wno-unused-function -I$M8/include -I$PB/include -I$SRC -pthread -o $BIN/dec_1fg $SRC/dec_1fg.cpp $LIBZ
apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -lineinfo -I$ND/include -I$SRC -Xcompiler -Wall,-Wno-unused-function,-pthread,-msse4.2 \
  -o $BIN/read_1fg $SRC/read_1fg.cu -L$ND/lib64 -lnvcomp -Xlinker -rpath,$ND/lib64
ls -l $BIN
