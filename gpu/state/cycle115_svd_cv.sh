#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-115 — Speckled cross-validation for truncated SVD (mirror of NMF CV).
# Builds only reduce_svd_cv_correctness; runs three gtest cases.

#SBATCH --job-name=sg_c115_svd_cv
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle115_svd_cv_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-115: SVD Cross-Validation Verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

# Incremental rebuild: only the new SVD CV test binary.
# Always reconfigure so CMake picks up the new target from CMakeLists.txt.
# (Incremental configure is fast; build is incremental automatically.)
echo "--- cmake configure (picks up new reduce_svd_cv_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build reduce_svd_cv_correctness ---"
cmake --build "$BUILD_DIR" \
    --target reduce_svd_cv_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

SVD_CV_BIN="$BUILD_DIR/tests/reduce_svd_cv_correctness"

# --- Smoke test ---
echo ""
echo "--- Smoke (200×500 synthetic, k={5,10,30,100}) ---"
"$SVD_CV_BIN" --gtest_color=no --gtest_filter="SvdCvTest.Smoke_200x500" 2>&1
SMOKE_EXIT=$?
echo "smoke_exit=$SMOKE_EXIT"

# --- Rank-recovery test ---
echo ""
echo "--- Rank-recovery (rank-10 ground truth) ---"
"$SVD_CV_BIN" --gtest_color=no --gtest_filter="SvdCvTest.RankRecovery_TrueRank10" 2>&1
RANK_EXIT=$?
echo "rank_exit=$RANK_EXIT"

# --- Determinism test ---
echo ""
echo "--- Determinism (rel_err < 1e-4) ---"
"$SVD_CV_BIN" --gtest_color=no --gtest_filter="SvdCvTest.Determinism_SameSeed" 2>&1
DET_EXIT=$?
echo "det_exit=$DET_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                              $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Smoke (200×500):                    $([ $SMOKE_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Rank-recovery (true rank=10):       $([ $RANK_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Determinism (rel_err < 1e-4):       $([ $DET_EXIT   -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
[ $SMOKE_EXIT -ne 0 ] && OVERALL=1
[ $RANK_EXIT  -ne 0 ] && OVERALL=1
[ $DET_EXIT   -ne 0 ] && OVERALL=1
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
