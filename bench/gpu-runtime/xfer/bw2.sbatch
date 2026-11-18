#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 16 --mem=96G --time=00:20:00 -w g052 -J xfer-bw2 -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/gpu-runtime/xfer
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc); echo nvcc=$NVCC
cd $D && $NVCC -O2 -std=c++14 -arch=sm_90 -o bw2 bw2.cu -lpthread 2>&1 | tail -5 && ./bw2 8
