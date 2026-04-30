#!/bin/bash
#SBATCH --job-name=sg_c86d2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_h100_nvl
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle86_d_verify2_%j.log

# ============================================================
# Cycle 86 Phase D — OPTIM-NMF-K50 self-verification gate
# Rebuild from build_cycle86_verify (fresh dir) with fixed fit.h
# ============================================================

set -e
export PATH=/opt/rh/gcc-toolset-13/root/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== DATE: $(date) ==="
echo ""

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle86_verify
SOURCE_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

# --- Step 1: Rebuild ---
echo "=== STEP 1: cmake --build $BUILD_DIR -j ==="
cmake --build "$BUILD_DIR" -j 16 2>&1
echo "BUILD EXIT: $?"
echo ""

# --- Step 2: ctest with label filter "nmf" (targets correctness tests) ---
echo "=== STEP 2: ctest -L nmf --output-on-failure ==="
ctest --test-dir "$BUILD_DIR" -L "nmf" --output-on-failure 2>&1
CTEST_EXIT=$?
echo "CTEST EXIT: $CTEST_EXIT"
echo ""

# --- Step 3: Run the Phase B profile driver (direct factornet call — shows original regression) ---
echo "=== STEP 3: bench_nmf_profile_c86 (direct factornet, shows regression + MU control) ==="
PROFILE_BIN="$BUILD_DIR/bench/bench_nmf_profile_c86"
if [ -f "$PROFILE_BIN" ]; then
    "$PROFILE_BIN" 2>&1
else
    echo "SKIP: profile binary not in build_cycle86_verify; checking build_cycle86_profile..."
    PROFILE_BIN="$SOURCE_DIR/build_cycle86_profile/bench/bench_nmf_profile_c86"
    if [ -f "$PROFILE_BIN" ]; then
        "$PROFILE_BIN" 2>&1
    else
        echo "SKIP: profile binary not found"
    fi
fi
echo ""

echo "=== DONE: $(date) ==="
