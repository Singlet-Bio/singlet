#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 112 — capture HvgTest.Tiny_SeuratV3_Top50 assertion in full (no tail).

#SBATCH --job-name=sg_c112_seurat
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle112_seurat_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== Cycle 112: HvgTest.Tiny_SeuratV3_Top50 targeted ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

# Rebuild from previous build dir (incremental, fast)
if [ -d "$BUILD_DIR" ]; then
    echo "--- incremental rebuild ---"
    cmake --build "$BUILD_DIR" --target preprocess_hvg_correctness -j8 2>&1 | tail -5
else
    echo "build dir missing, configure+build from scratch"
    cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
        -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
        -DSINGLET_GPU_BUILD_TESTS=ON \
        -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -10
    cmake --build "$BUILD_DIR" --target preprocess_hvg_correctness -j8 2>&1 | tail -5
fi

HVG_BIN="$BUILD_DIR/tests/preprocess_hvg_correctness"
echo ""
echo "--- $HVG_BIN --gtest_filter=HvgTest.Tiny_SeuratV3_Top50 ---"
"$HVG_BIN" --gtest_color=no --gtest_filter=HvgTest.Tiny_SeuratV3_Top50 2>&1
EXIT=$?
echo ""
echo "EXIT=$EXIT"

# Also run the LARGE-scale Seurat v3 test for comparison (it passed in 368732)
echo ""
echo "--- $HVG_BIN --gtest_filter=HvgTest.Gsm4037629_SeuratV3_Top2000 ---"
"$HVG_BIN" --gtest_color=no --gtest_filter=HvgTest.Gsm4037629_SeuratV3_Top2000 2>&1 | tail -25
EXIT2=$?
echo "EXIT2=$EXIT2"

date
