#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-138 — Symphony (Kang et al. 2021) reference-based cell-type mapping on GPU.
# Reference PCA projection + soft cluster assignment (inverse-distance weighting)
# + label transfer via cuBLAS Sgemm + custom standardize/distance/softassign kernels.
# Kang JB, Nathan A, Weinand K, et al. (2021) Nat Commun 12:5890.
# Builds only anno_symphony_correctness; runs 5 gtest cases.

#SBATCH --job-name=symphony_c138
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle138_symphony_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-138: Symphony correctness verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

echo "--- cmake configure (picks up new anno_symphony_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build anno_symphony_correctness ---"
cmake --build "$BUILD_DIR" \
    --target anno_symphony_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

SY_BIN="$BUILD_DIR/tests/anno_symphony_correctness"

echo ""
echo "--- Test 1: Symphony_TinyClosedForm ---"
"$SY_BIN" --gtest_color=no \
    --gtest_filter="SymphonyFixture.Symphony_TinyClosedForm" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

echo ""
echo "--- Test 2: Symphony_QueryNearCluster_AssignsClusterLabel ---"
"$SY_BIN" --gtest_color=no \
    --gtest_filter="SymphonyFixture.Symphony_QueryNearCluster_AssignsClusterLabel" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

echo ""
echo "--- Test 3: Symphony_AmbiguousQuery_LowConfidence ---"
"$SY_BIN" --gtest_color=no \
    --gtest_filter="SymphonyFixture.Symphony_AmbiguousQuery_LowConfidence" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

echo ""
echo "--- Test 4: Symphony_Determinism_BitIdentical ---"
"$SY_BIN" --gtest_color=no \
    --gtest_filter="SymphonyFixture.Symphony_Determinism_BitIdentical" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

echo ""
echo "--- Test 5: Symphony_DegenerateZeroSigma_Survives ---"
"$SY_BIN" --gtest_color=no \
    --gtest_filter="SymphonyFixture.Symphony_DegenerateZeroSigma_Survives" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                                       $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1  Symphony_TinyClosedForm:                             $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2  Symphony_QueryNearCluster_AssignsClusterLabel:       $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3  Symphony_AmbiguousQuery_LowConfidence:               $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4  Symphony_Determinism_BitIdentical:                   $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5  Symphony_DegenerateZeroSigma_Survives:               $([ $T5_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
for EXIT in $T1_EXIT $T2_EXIT $T3_EXIT $T4_EXIT $T5_EXIT; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
