#!/bin/bash
#SBATCH --job-name=w11-pu-compile
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# compile pileup.cu (sm_90, -fmad=false) in the CUDA devel image and extract.cpp with the pbench toolchain; no GPU needed
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; M=$W/jobs/w11/pileup; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; ENV=$W/envs/pbench
mkdir -p $M/bin
apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -o $M/bin/pileup $M/gpu/pileup/pileup.cu 2>&1 | grep -v "Wno-deprecated" | head -80
ls -la $M/bin/pileup
$ENV/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -I$ENV/include -I$M/gpu/pileup -o $M/bin/extract $M/gpu/pileup/extract.cpp -L$ENV/lib -lhts -Wl,-rpath,$ENV/lib 2>&1 | head -40
ls -la $M/bin/extract
