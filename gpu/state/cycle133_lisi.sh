#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-133 — LISI (Local Inverse Simpson's Index) per-cell batch/cluster metric.
# GPU-native LISI: singlet_gpu::integrate::lisi().
# Builds only integrate_lisi_correctness; runs 5 gtest cases.

#SBATCH --job-name=lisi_c133
#SBATCH --partition=gpu
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle133_lisi_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-133: LISI correctness verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

echo "--- cmake configure (picks up new integrate_lisi_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build integrate_lisi_correctness ---"
cmake --build "$BUILD_DIR" \
    --target integrate_lisi_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

LISI_BIN="$BUILD_DIR/tests/integrate_lisi_correctness"

echo ""
echo "--- Test 1: Lisi_PerfectMixing_HighScore (100 cells, k=10, 2 labels, 5/5 split → LISI=2.0) ---"
"$LISI_BIN" --gtest_color=no \
    --gtest_filter="LisiTest.Lisi_PerfectMixing_HighScore" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

echo ""
echo "--- Test 2: Lisi_NoMixing_LowScore (100 cells, k=10, 2 labels, all same → LISI=1.0) ---"
"$LISI_BIN" --gtest_color=no \
    --gtest_filter="LisiTest.Lisi_NoMixing_LowScore" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

echo ""
echo "--- Test 3: Lisi_FourBatches_BalancedMaxLISI (200 cells, k=20, 4 labels, 5-5-5-5 → LISI=4.0) ---"
"$LISI_BIN" --gtest_color=no \
    --gtest_filter="LisiTest.Lisi_FourBatches_BalancedMaxLISI" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

echo ""
echo "--- Test 4: Lisi_Determinism_BitIdentical (80 cells, k=10, 3 labels, rel_err=0) ---"
"$LISI_BIN" --gtest_color=no \
    --gtest_filter="LisiTest.Lisi_Determinism_BitIdentical" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

echo ""
echo "--- Test 5: Lisi_SingleLabel_DegenerateCase (50 cells, k=8, n_labels=1, LISI=1.0, no NaN/Inf) ---"
"$LISI_BIN" --gtest_color=no \
    --gtest_filter="LisiTest.Lisi_SingleLabel_DegenerateCase" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                                    $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1  Lisi_PerfectMixing_HighScore:                     $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2  Lisi_NoMixing_LowScore:                           $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3  Lisi_FourBatches_BalancedMaxLISI:                 $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4  Lisi_Determinism_BitIdentical:                    $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5  Lisi_SingleLabel_DegenerateCase:                  $([ $T5_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
for EXIT in $T1_EXIT $T2_EXIT $T3_EXIT $T4_EXIT $T5_EXIT; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
