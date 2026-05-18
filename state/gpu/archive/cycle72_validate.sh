#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 72 validation job — BUG-WILCOXON-POST-NORMALIZE-CRASH regression test.
# Rebuilds de_wilcoxon_correctness from the repo as-is, then runs only Test83.
#
# Log: singlet-gpu/state/cycle72_validate_${SLURM_JOB_ID}.log
# Design doc: singlet-gpu/state/designs/72-wilcoxon-postnorm-crash.md

#SBATCH --job-name=cycle72-wilcoxon-postnorm
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle72_validate_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle72_validate_%j.log

set -euo pipefail

REPO=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR="${REPO}/build"

echo "=== Cycle 72 validation ==="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Date         : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repo         : ${REPO}"
echo "Build dir    : ${BUILD_DIR}"
echo ""

# --- GPU sanity ---
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# --- Check lognorm fix status before build ---
echo "=== lognorm.h sync check ==="
LOGNORM="${REPO}/include/singlet-gpu/preprocess/lognorm.h"
if grep -n "cudaStreamSynchronize(stream)" "${LOGNORM}" | grep -v "// wait for t\[\]"; then
    echo "FIX STATUS: cudaStreamSynchronize(stream) present in lognorm.h (post-fix)"
else
    echo "FIX STATUS: cudaStreamSynchronize(stream) NOT found near return (pre-fix — crash expected)"
fi
echo ""

# --- Build ---
echo "=== Build ==="
cd "${BUILD_DIR}"

# Rebuild only the wilcoxon correctness target (fast, avoids full rebuild).
cmake --build . --target de_wilcoxon_correctness -j8 2>&1
BUILD_EXIT=$?
echo "Build exit code: ${BUILD_EXIT}"
if [ "${BUILD_EXIT}" -ne 0 ]; then
    echo "BUILD FAILED — cannot run test"
    exit "${BUILD_EXIT}"
fi
echo ""

# --- Run targeted ctest ---
echo "=== ctest Test83_RealDataSized_PostNormalize_NoCrash ==="
set +e
ctest -R "Test83_RealDataSized_PostNormalize_NoCrash" \
      --output-on-failure \
      --timeout 600 \
      2>&1
CTEST_EXIT=$?
set -e

echo ""
echo "=== ctest exit code: ${CTEST_EXIT} ==="
if [ "${CTEST_EXIT}" -eq 0 ]; then
    echo "RESULT: PASS"
else
    echo "RESULT: FAIL (exit ${CTEST_EXIT})"
    echo "NOTE: if exit code is non-zero and log shows 'illegal memory access',"
    echo "      that is the expected pre-fix behavior for BUG-WILCOXON-POST-NORMALIZE-CRASH."
    echo "      The fix (cudaStreamSynchronize before return in lognorm.h) has not yet landed."
fi

echo ""
echo "=== Done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit "${CTEST_EXIT}"
