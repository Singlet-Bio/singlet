#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-147 — Scrublet audit: build + run existing qc/doublet_score.h tests.
# Cycle 13 vintage; 5 tests in qc_doublet_correctness.cpp.

#SBATCH --job-name=sg_c147_scrub
#SBATCH --partition=gpu
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle147_scrublet_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2

echo "=== CYCLE-147: Scrublet audit ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"

cmake -S /mnt/home/debruinz/Singlet-AI/singlet-gpu -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -10

echo ""
echo "--- build qc_doublet_correctness ---"
cmake --build "$BUILD_DIR" --target qc_doublet_correctness -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "build_exit=$BUILD_EXIT"

if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — exit"
    exit 1
fi

BIN="$BUILD_DIR/tests/qc_doublet_correctness"

echo ""
echo "--- run scrublet doublet tests ---"
"$BIN" --gtest_color=no 2>&1
RUN_EXIT=$?
echo "run_exit=$RUN_EXIT"

echo ""
echo "=== SUMMARY ==="
echo "Build: $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
echo "Run:   $([ $RUN_EXIT -eq 0 ] && echo PASS || echo FAIL)"
date
exit $RUN_EXIT
