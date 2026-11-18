#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 16 --mem=96G --time=00:20:00 -w g052 -J xfer-bw -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/gpu-runtime/xfer
hostname; nvidia-smi --query-gpu=name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
nvidia-smi -q | grep -A3 -i "link" | head -40
ls /usr/local/cuda/extras/demo_suite/bandwidthTest 2>&1
for m in pinned pageable; do /usr/local/cuda/extras/demo_suite/bandwidthTest --memory=$m --mode=shmoo 2>&1 | tail -8; done
NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc); echo nvcc=$NVCC
cd $D && $NVCC -O2 -std=c++14 -arch=sm_90 -o bw bw.cu -lpthread 2>&1 | tail -5 && ./bw 8
