#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 113 — deviance HVG correctness fix verify.
#
# Root causes fixed in this cycle (hvg.h):
#   1. result.deviance was set to d_dev_s (sorted descending) instead of
#      d_deviance (gene-indexed, original order) → Spearman ≈ 0.05 for all
#      non-degenerate tests (job 368732).
#   2. out_top was allocated at top_n (clamped to m) but copy_dev_result_hvg
#      copied cfg.top_n entries → CUDA_ERROR_INVALID_VALUE ("invalid argument")
#      when cfg.top_n > n_genes (streaming test D5: 2000 > 500).
#
# Expected: 9/9 deviance tests PASS (was 2/9).

#SBATCH --job-name=sg_c113_deviance
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle113_deviance_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== Cycle 113: deviance HVG correctness fix ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

# Incremental rebuild (hvg.h changed; only deviance targets need recompile).
if [ -d "$BUILD_DIR" ]; then
    echo "--- incremental rebuild ---"
    cmake --build "$BUILD_DIR" \
        --target preprocess_hvg_correctness preprocess_hvg_deviance_correctness \
        -j8 2>&1 | tail -10
else
    echo "--- cold configure + build ---"
    cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
        -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
        -DSINGLET_GPU_BUILD_TESTS=ON \
        -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -15
    cmake --build "$BUILD_DIR" \
        --target preprocess_hvg_correctness preprocess_hvg_deviance_correctness \
        -j8 2>&1 | tail -10
fi

echo ""
echo "Build exit: $?"

HVG_BIN="$BUILD_DIR/tests/preprocess_hvg_correctness"
DEV_BIN="$BUILD_DIR/tests/preprocess_hvg_deviance_correctness"

# ---- preprocess_hvg_correctness: all 9 deviance sub-tests ----
echo ""
echo "=== preprocess_hvg_correctness: DevianceHvgTest.* ==="
"$HVG_BIN" --gtest_color=no \
    --gtest_filter="DevianceHvgTest.*" 2>&1
EXIT1=$?
echo "EXIT1=$EXIT1"

# ---- preprocess_hvg_deviance_correctness: dedicated binary ----
echo ""
echo "=== preprocess_hvg_deviance_correctness: Test4/5/6 ==="
"$DEV_BIN" --gtest_color=no \
    --gtest_filter="DevianceHvgTest.Test4_PoissonNullAgreement:\
DevianceHvgTest.Test5_DeterminismIdempotent:\
DevianceHvgTest.Test6_TopNMaskConsistency" 2>&1
EXIT2=$?
echo "EXIT2=$EXIT2"

# ---- Regression guard: non-deviance HVG tests must not regress ----
echo ""
echo "=== regression guard: non-deviance HvgTest.* ==="
"$HVG_BIN" --gtest_color=no \
    --gtest_filter="HvgTest.*" 2>&1 | tail -20
EXIT3=$?
echo "EXIT3=$EXIT3"

echo ""
echo "=== Summary ==="
echo "DevianceHvgTest.*        exit=$EXIT1  (expect 0)"
echo "Test4/5/6 dedicated bin  exit=$EXIT2  (expect 0)"
echo "HvgTest regression       exit=$EXIT3  (expect 0)"
date
