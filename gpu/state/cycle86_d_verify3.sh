#!/bin/bash
#SBATCH --job-name=sg_c86d3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_h100_nvl
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle86_d_verify3_%j.log

# ============================================================
# Cycle 86 Phase D — OPTIM-NMF-K50 correctness gate
# Uses the already-built build_cycle86_profile which compiled clean (BUILD EXIT: 0)
# Runs NMF correctness tests directly + profile bench
# ============================================================

export PATH=/opt/rh/gcc-toolset-13/root/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== DATE: $(date) ==="
echo ""

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle86_profile

# --- Step 1: Run NMF correctness binary directly ---
echo "=== STEP 1: NMF correctness tests (direct binary) ==="
NMF_TEST="$BUILD_DIR/tests/reduce_nmf_correctness"
if [ -f "$NMF_TEST" ]; then
    "$NMF_TEST" 2>&1
    echo "NMF CORRECTNESS EXIT: $?"
else
    echo "BINARY NOT FOUND: $NMF_TEST"
fi
echo ""

# --- Step 2: ctest with LABEL filter (not regex) ==="
echo "=== STEP 2: ctest -L nmf (label filter for correctness tests) ==="
ctest --test-dir "$BUILD_DIR" -L "nmf" --output-on-failure 2>&1
echo "CTEST EXIT: $?"
echo ""

# --- Step 3: Profile bench to confirm wall time ==="
echo "=== STEP 3: bench_nmf_profile_c86 (k=50 wall time verification) ==="
PROFILE_BIN="$BUILD_DIR/bench/bench_nmf_profile_c86"
if [ -f "$PROFILE_BIN" ]; then
    "$PROFILE_BIN" 2>&1
else
    echo "SKIP: profile binary not found"
fi
echo ""

echo "=== DONE: $(date) ==="
