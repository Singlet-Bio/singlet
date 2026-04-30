#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-153 — Scrublet doublet-score focused fix (qc/doublet_score.h, cycle-13 vintage).
# Fixes: (1) find_knee_threshold placed threshold beyond doublet bump → zero calls (AUC 0.63)
#             fixed to valley-minimum + left-edge-of-bump detection.
#        (2) Spearman test used random projection vs scrublet's TruncatedSVD PCA → divergent
#             kNN graphs; fixed by exporting scrublet's PCA from reference script and using
#             the same embedding in both frameworks.
# Builds only qc_doublet_correctness; runs all 5 gtest cases.

#SBATCH --job-name=scrublet_c153
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g003
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle153_scrublet_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-153: Scrublet doublet-score correctness verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

echo "--- cmake configure (picks up updated qc/doublet_score.h) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build qc_doublet_correctness ---"
cmake --build "$BUILD_DIR" \
    --target qc_doublet_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

DOUBLET_BIN="$BUILD_DIR/tests/qc_doublet_correctness"

echo ""
echo "--- Test 1: Doublet_TinySynthetic_VsScrublet ---"
"$DOUBLET_BIN" --gtest_color=no \
    --gtest_filter="DoubletFixture.Doublet_TinySynthetic_VsScrublet" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

echo ""
echo "--- Test 2: Doublet_GSM4037629_RealData ---"
"$DOUBLET_BIN" --gtest_color=no \
    --gtest_filter="DoubletFixture.Doublet_GSM4037629_RealData" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

echo ""
echo "--- Test 3: Doublet_AutoThreshold_ROC ---"
"$DOUBLET_BIN" --gtest_color=no \
    --gtest_filter="DoubletFixture.Doublet_AutoThreshold_ROC" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

echo ""
echo "--- Test 4: Doublet_Determinism_BitIdentical ---"
"$DOUBLET_BIN" --gtest_color=no \
    --gtest_filter="DoubletFixture.Doublet_Determinism_BitIdentical" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

echo ""
echo "--- Test 5: Doublet_NSynth_Sensitivity ---"
"$DOUBLET_BIN" --gtest_color=no \
    --gtest_filter="DoubletFixture.Doublet_NSynth_Sensitivity" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                      $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1  Doublet_TinySynthetic_VsScrublet:   $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2  Doublet_GSM4037629_RealData:        $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3  Doublet_AutoThreshold_ROC:          $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4  Doublet_Determinism_BitIdentical:   $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5  Doublet_NSynth_Sensitivity:         $([ $T5_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
for EXIT in $T1_EXIT $T2_EXIT $T3_EXIT $T4_EXIT $T5_EXIT; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
