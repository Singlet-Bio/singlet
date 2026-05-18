#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-127 — Poisson-null HVG selection (modelGeneVarByPoisson).
# Lun ATL, McCarthy DJ, Marioni JC (2016) F1000Research 5:2122.
# Builds only preprocess_model_gene_var_correctness; runs five gtest cases.

#SBATCH --job-name=sg_c127_mgv
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle127_modelgenevar_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-127: modelGeneVarByPoisson HVG Verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

# Incremental rebuild: only the new model_gene_var test binary.
# Always reconfigure so CMake picks up the new target from CMakeLists.txt.
echo "--- cmake configure (picks up new preprocess_model_gene_var_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build preprocess_model_gene_var_correctness ---"
cmake --build "$BUILD_DIR" \
    --target preprocess_model_gene_var_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

MGV_BIN="$BUILD_DIR/tests/preprocess_model_gene_var_correctness"

# --- Test 1: Tiny closed-form vs CPU reference ---
echo ""
echo "--- Test 1: Tiny_Closed_Form (5x8, abs_err<1e-5 mean, <1e-3 var) ---"
"$MGV_BIN" --gtest_color=no \
    --gtest_filter="ModelGeneVarTest.Tiny_Closed_Form" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

# --- Test 2: Pure Poisson background has zero bio_var ---
echo ""
echo "--- Test 2: Pure_Poisson_Background_Has_Zero_BioVar (100x500, median<0.5) ---"
"$MGV_BIN" --gtest_color=no \
    --gtest_filter="ModelGeneVarTest.Pure_Poisson_Background_Has_Zero_BioVar" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

# --- Test 3: Planted high-var genes recovered in top-10 ---
echo ""
echo "--- Test 3: Planted_HighVar_Genes_Recovered (50x500, >=8/10 recall) ---"
"$MGV_BIN" --gtest_color=no \
    --gtest_filter="ModelGeneVarTest.Planted_HighVar_Genes_Recovered" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

# --- Test 4: Determinism (rel_err < 1e-4) ---
echo ""
echo "--- Test 4: Determinism_Same_Input (30x200, rel_err<1e-4) ---"
"$MGV_BIN" --gtest_color=no \
    --gtest_filter="ModelGeneVarTest.Determinism_Same_Input" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

# --- Test 5: min_mean filter ---
echo ""
echo "--- Test 5: Min_Mean_Filter (50x300, low-mean genes excluded) ---"
"$MGV_BIN" --gtest_color=no \
    --gtest_filter="ModelGeneVarTest.Min_Mean_Filter" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                          $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1 Tiny_Closed_Form:                        $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2 Pure_Poisson_Background_Has_Zero_BioVar: $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3 Planted_HighVar_Genes_Recovered:         $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4 Determinism_Same_Input:                  $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5 Min_Mean_Filter:                         $([ $T5_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
[ $T1_EXIT -ne 0 ] && OVERALL=1
[ $T2_EXIT -ne 0 ] && OVERALL=1
[ $T3_EXIT -ne 0 ] && OVERALL=1
[ $T4_EXIT -ne 0 ] && OVERALL=1
[ $T5_EXIT -ne 0 ] && OVERALL=1
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
