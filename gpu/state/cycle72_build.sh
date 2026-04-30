#!/bin/bash
#SBATCH --job-name=cycle72_build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle72_build_%j.log

set -e

cd /mnt/home/debruinz/Singlet-AI/singlet-gpu

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -30
cmake --build build -j8 --target de_wilcoxon_correctness de_ttest_correctness de_donor_pseudobulk_correctness lognorm_correctness 2>&1 | tail -40
echo "BUILD_EXIT=$?"
