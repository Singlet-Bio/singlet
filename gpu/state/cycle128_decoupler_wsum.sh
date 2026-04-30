#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-128 — decoupleR WSUM + WMEAN pathway-scoring methods.
# Badia-i-Mompel P et al. (2022) Bioinformatics Advances 2:vbac016.
# Builds only enrich_decoupler_wsum_correctness; runs 10 gtest cases.

#SBATCH --job-name=sg_c128_wsum
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle128_decoupler_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-128: decoupleR WSUM + WMEAN pathway-scoring Verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

# Incremental rebuild: only the new enrich_decoupler_wsum_correctness target.
# Always reconfigure so CMake picks up the new target from CMakeLists.txt.
echo "--- cmake configure (picks up new enrich_decoupler_wsum_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build enrich_decoupler_wsum_correctness ---"
cmake --build "$BUILD_DIR" \
    --target enrich_decoupler_wsum_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

WSUM_BIN="$BUILD_DIR/tests/enrich_decoupler_wsum_correctness"

# --- WSUM Tests ---

echo ""
echo "--- Test 1: Wsum_TinyClosedForm (5x8x3, abs_err<1e-4) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WsumTest.Wsum_TinyClosedForm" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

echo ""
echo "--- Test 2: Wsum_RealMatrices_VsCpu (50x100x5, abs_err<1e-3) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WsumTest.Wsum_RealMatrices_VsCpu" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

echo ""
echo "--- Test 3: Wsum_AllZerosWeights_AllZerosScores (20x30x4) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WsumTest.Wsum_AllZerosWeights_AllZerosScores" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

echo ""
echo "--- Test 4: Wsum_Determinism_SameInput (30x60x4, rel_err<1e-4) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WsumTest.Wsum_Determinism_SameInput" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

echo ""
echo "--- Test 5: Wsum_XScale_PropagatesScore (15x20x3) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WsumTest.Wsum_XScale_PropagatesScore" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

# --- WMEAN Tests ---

echo ""
echo "--- Test 6: Wmean_TinyClosedForm (5x8x3, abs_err<1e-4) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WmeanTest.Wmean_TinyClosedForm" 2>&1
T6_EXIT=$?
echo "t6_exit=$T6_EXIT"

echo ""
echo "--- Test 7: Wmean_RealMatrices_VsCpu (50x100x5, abs_err<1e-3) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WmeanTest.Wmean_RealMatrices_VsCpu" 2>&1
T7_EXIT=$?
echo "t7_exit=$T7_EXIT"

echo ""
echo "--- Test 8: Wmean_AllZerosWeights_AllZerosScores (20x30x4) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WmeanTest.Wmean_AllZerosWeights_AllZerosScores" 2>&1
T8_EXIT=$?
echo "t8_exit=$T8_EXIT"

echo ""
echo "--- Test 9: Wmean_Determinism_SameInput (30x60x4, rel_err<1e-4) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WmeanTest.Wmean_Determinism_SameInput" 2>&1
T9_EXIT=$?
echo "t9_exit=$T9_EXIT"

echo ""
echo "--- Test 10: Wmean_WScale_PropagatesScore (15x20x3) ---"
"$WSUM_BIN" --gtest_color=no \
    --gtest_filter="WmeanTest.Wmean_WScale_PropagatesScore" 2>&1
T10_EXIT=$?
echo "t10_exit=$T10_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                      $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1  Wsum_TinyClosedForm:                $([ $T1_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2  Wsum_RealMatrices_VsCpu:            $([ $T2_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3  Wsum_AllZerosWeights_AllZerosScores:$([ $T3_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4  Wsum_Determinism_SameInput:         $([ $T4_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5  Wsum_XScale_PropagatesScore:        $([ $T5_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 6  Wmean_TinyClosedForm:               $([ $T6_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 7  Wmean_RealMatrices_VsCpu:           $([ $T7_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 8  Wmean_AllZerosWeights_AllZerosScores:$([ $T8_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 9  Wmean_Determinism_SameInput:        $([ $T9_EXIT  -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 10 Wmean_WScale_PropagatesScore:       $([ $T10_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
for EXIT in $T1_EXIT $T2_EXIT $T3_EXIT $T4_EXIT $T5_EXIT \
            $T6_EXIT $T7_EXIT $T8_EXIT $T9_EXIT $T10_EXIT; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
