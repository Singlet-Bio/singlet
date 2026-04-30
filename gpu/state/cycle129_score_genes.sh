#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-129 — scanpy.tl.score_genes per-cell gene-set scoring.
# Satija R et al. (2015) Nat Biotechnol 33:495-502 / Seurat AddModuleScore.
# Builds only enrich_score_genes_correctness; runs 5 gtest cases.

#SBATCH --job-name=sg_c129_scoregenes
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle129_score_genes_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-129: score_genes per-cell gene-set scoring Verify ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

# Incremental rebuild: only the new enrich_score_genes_correctness target.
# Always reconfigure so CMake picks up the new target from CMakeLists.txt.
echo "--- cmake configure (picks up new enrich_score_genes_correctness target) ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20

echo "--- build enrich_score_genes_correctness ---"
cmake --build "$BUILD_DIR" \
    --target enrich_score_genes_correctness \
    -j8 2>&1
BUILD_EXIT=$?

echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

SG_BIN="$BUILD_DIR/tests/enrich_score_genes_correctness"

echo ""
echo "--- Test 1: ScoreGenes_TinyClosedForm (10x20x1, abs_err<1e-4) ---"
"$SG_BIN" --gtest_color=no \
    --gtest_filter="ScoreGenesTest.ScoreGenes_TinyClosedForm" 2>&1
T1_EXIT=$?
echo "t1_exit=$T1_EXIT"

echo ""
echo "--- Test 2: ScoreGenes_PlantedSet_HighScore (50x200, delta>=1.0) ---"
"$SG_BIN" --gtest_color=no \
    --gtest_filter="ScoreGenesTest.ScoreGenes_PlantedSet_HighScore" 2>&1
T2_EXIT=$?
echo "t2_exit=$T2_EXIT"

echo ""
echo "--- Test 3: ScoreGenes_MultipleSets_Independent (50x100x3) ---"
"$SG_BIN" --gtest_color=no \
    --gtest_filter="ScoreGenesTest.ScoreGenes_MultipleSets_Independent" 2>&1
T3_EXIT=$?
echo "t3_exit=$T3_EXIT"

echo ""
echo "--- Test 4: ScoreGenes_Determinism_SameSeed (30x50x2) ---"
"$SG_BIN" --gtest_color=no \
    --gtest_filter="ScoreGenesTest.ScoreGenes_Determinism_SameSeed" 2>&1
T4_EXIT=$?
echo "t4_exit=$T4_EXIT"

echo ""
echo "--- Test 5: ScoreGenes_AllOnesInput (20x40x1, scores~0) ---"
"$SG_BIN" --gtest_color=no \
    --gtest_filter="ScoreGenesTest.ScoreGenes_AllOnesInput" 2>&1
T5_EXIT=$?
echo "t5_exit=$T5_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build:                                        $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 1  ScoreGenes_TinyClosedForm:            $([ $T1_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 2  ScoreGenes_PlantedSet_HighScore:      $([ $T2_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 3  ScoreGenes_MultipleSets_Independent:  $([ $T3_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 4  ScoreGenes_Determinism_SameSeed:      $([ $T4_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Test 5  ScoreGenes_AllOnesInput:              $([ $T5_EXIT -eq 0 ] && echo PASS || echo FAIL)"

OVERALL=0
for EXIT in $T1_EXIT $T2_EXIT $T3_EXIT $T4_EXIT $T5_EXIT; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
