#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-136 — decoupleR MLM (Multivariate Linear Model) pathway scoring on GPU.
# Jointly regresses all pathways on each cell via closed-form OLS:
#   B = (W^T W)^{-1} W^T X
# cuBLAS Sgemm (W^T W) + cuSPARSE SpMM (W^T X) + cuSOLVER Potrf/Potrs + cuBLAS Sgeam.
# Badia-i-Mompel et al. (2022) Bioinformatics Advances 2:vbac016.
# Builds only enrich_decoupler_mlm_correctness; runs 5 gtest cases.

#SBATCH --job-name=mlm_c136
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle136_mlm_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-136: MLM correctness verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

echo "--- cmake configure (picks up new enrich_decoupler_mlm_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build enrich_decoupler_mlm_correctness ---"
cmake --build "$BUILD_DIR" \
    --target enrich_decoupler_mlm_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

MLM_BIN="$BUILD_DIR/tests/enrich_decoupler_mlm_correctness"

echo ""
echo "--- Test 1: Mlm_TinyClosedForm ---"
"$MLM_BIN" --gtest_color=no \
    --gtest_filter="MlmTest.Mlm_TinyClosedForm" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

echo ""
echo "--- Test 2: Mlm_OrthogonalPathways_MatchRef ---"
"$MLM_BIN" --gtest_color=no \
    --gtest_filter="MlmTest.Mlm_OrthogonalPathways_MatchRef" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

echo ""
echo "--- Test 3: Mlm_CorrelatedPathways_DiffersFromULM ---"
"$MLM_BIN" --gtest_color=no \
    --gtest_filter="MlmTest.Mlm_CorrelatedPathways_DiffersFromULM" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

echo ""
echo "--- Test 4: Mlm_Determinism_BitIdentical ---"
"$MLM_BIN" --gtest_color=no \
    --gtest_filter="MlmTest.Mlm_Determinism_BitIdentical" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

echo ""
echo "--- Test 5: Mlm_RankDeficient_RidgeStabilizes ---"
"$MLM_BIN" --gtest_color=no \
    --gtest_filter="MlmTest.Mlm_RankDeficient_RidgeStabilizes" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                                  $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1  Mlm_TinyClosedForm:                             $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2  Mlm_OrthogonalPathways_MatchRef:                $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3  Mlm_CorrelatedPathways_DiffersFromULM:          $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4  Mlm_Determinism_BitIdentical:                   $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5  Mlm_RankDeficient_RidgeStabilizes:              $([ $T5_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
for EXIT in $T1_EXIT $T2_EXIT $T3_EXIT $T4_EXIT $T5_EXIT; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
