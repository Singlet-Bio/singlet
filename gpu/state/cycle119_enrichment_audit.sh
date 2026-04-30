#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-119 — Enrichment frontier audit (post-CYCLE-105 factornet purge).
# Builds + runs gsea/aucell + gsea/fgsea + enrich/ssgsea + enrich/progeny tests.
# Goal: certify these 17 tests still pass after the factornet removal, or surface
# the regressions so we can prioritize fixes.

#SBATCH --job-name=sg_c119_enrich
#SBATCH --partition=gpu
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle119_enrich_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-119: Enrichment Frontier Audit ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
echo "Tests: AUCell(5) + fgsea(6) + ssgsea/progeny(6) = 17 cases"

echo "--- cmake configure ---"
cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -10

echo ""
echo "--- build gsea_aucell_correctness ---"
cmake --build "$BUILD_DIR" --target gsea_aucell_correctness -j8 2>&1 | tail -30
AUCELL_BUILD=$?
echo "aucell_build_exit=$AUCELL_BUILD"

echo ""
echo "--- build gsea_fgsea_correctness ---"
cmake --build "$BUILD_DIR" --target gsea_fgsea_correctness -j8 2>&1 | tail -30
FGSEA_BUILD=$?
echo "fgsea_build_exit=$FGSEA_BUILD"

echo ""
echo "--- build enrich_ssgsea_progeny_correctness ---"
cmake --build "$BUILD_DIR" --target enrich_ssgsea_progeny_correctness -j8 2>&1 | tail -30
ENRICH_BUILD=$?
echo "enrich_build_exit=$ENRICH_BUILD"

# --- Run tests ---
AUCELL_BIN="$BUILD_DIR/tests/gsea_aucell_correctness"
FGSEA_BIN="$BUILD_DIR/tests/gsea_fgsea_correctness"
ENRICH_BIN="$BUILD_DIR/tests/enrich_ssgsea_progeny_correctness"

AUCELL_RUN=99
FGSEA_RUN=99
ENRICH_RUN=99

if [ $AUCELL_BUILD -eq 0 ] && [ -x "$AUCELL_BIN" ]; then
    echo ""
    echo "--- run AUCell tests ---"
    "$AUCELL_BIN" --gtest_color=no 2>&1
    AUCELL_RUN=$?
    echo "aucell_run_exit=$AUCELL_RUN"
fi

if [ $FGSEA_BUILD -eq 0 ] && [ -x "$FGSEA_BIN" ]; then
    echo ""
    echo "--- run fgsea tests ---"
    "$FGSEA_BIN" --gtest_color=no 2>&1
    FGSEA_RUN=$?
    echo "fgsea_run_exit=$FGSEA_RUN"
fi

if [ $ENRICH_BUILD -eq 0 ] && [ -x "$ENRICH_BIN" ]; then
    echo ""
    echo "--- run ssgsea/progeny tests ---"
    "$ENRICH_BIN" --gtest_color=no 2>&1
    ENRICH_RUN=$?
    echo "enrich_run_exit=$ENRICH_RUN"
fi

echo ""
echo "=== SUMMARY ==="
echo "Build AUCell:   $([ $AUCELL_BUILD -eq 0 ] && echo PASS || echo FAIL)"
echo "Build fgsea:    $([ $FGSEA_BUILD  -eq 0 ] && echo PASS || echo FAIL)"
echo "Build enrich:   $([ $ENRICH_BUILD -eq 0 ] && echo PASS || echo FAIL)"
echo "Run   AUCell:   $([ $AUCELL_RUN -eq 0 ] && echo PASS || echo "FAIL ($AUCELL_RUN)")"
echo "Run   fgsea:    $([ $FGSEA_RUN  -eq 0 ] && echo PASS || echo "FAIL ($FGSEA_RUN)")"
echo "Run   enrich:   $([ $ENRICH_RUN -eq 0 ] && echo PASS || echo "FAIL ($ENRICH_RUN)")"

OVERALL=0
[ $AUCELL_BUILD -ne 0 ] && OVERALL=1
[ $FGSEA_BUILD  -ne 0 ] && OVERALL=1
[ $ENRICH_BUILD -ne 0 ] && OVERALL=1
[ $AUCELL_RUN   -ne 0 ] && OVERALL=1
[ $FGSEA_RUN    -ne 0 ] && OVERALL=1
[ $ENRICH_RUN   -ne 0 ] && OVERALL=1
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
