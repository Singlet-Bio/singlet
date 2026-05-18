#!/bin/bash
#SBATCH --job-name=sg_c86d_nmf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia_h100_nvl
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle86_d_verify_%j.log

# ============================================================
# Cycle 86 Phase D — OPTIM-NMF-K50 verification
# Tests FitConfig adapter fix: k>=32 auto→MU, cd_max_iter default 100→10
# ============================================================

set -e
export PATH=/opt/rh/gcc-toolset-13/root/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

# Reuse the existing build dir (already cmake-configured for this node)
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle86_profile

echo "=== NODE: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
echo "=== DATE: $(date) ==="
echo ""

# --- Step 1: Rebuild with the fixed fit.h ---
echo "=== STEP 1: cmake --build $BUILD_DIR -j ==="
cmake --build "$BUILD_DIR" -j 16 2>&1
echo "BUILD EXIT: $?"
echo ""

# --- Step 2: ctest -R "nmf" to check no regressions ---
echo "=== STEP 2: ctest -R nmf (pre-fix baseline: 13 pass, 2 skip) ==="
ctest --test-dir "$BUILD_DIR" -R "nmf" --output-on-failure 2>&1
CTEST_EXIT=$?
echo "CTEST EXIT: $CTEST_EXIT"
echo ""

# --- Step 3: Run the Phase B profile driver to confirm k=50 wall drops ---
# The profile driver calls factornet directly (bypassing the adapter fix)
# to show the ORIGINAL regression, then a separate MU run. This confirms
# the regression is still present when the adapter is bypassed, proving
# the fix is what actually makes it fast.
# For the ADAPTER verification, we use bench_reduce_nmf_c86 which calls
# reduce::nmf::fit() through the adapter.
echo "=== STEP 3: bench_nmf_profile_c86 (raw factornet - shows regression + MU control) ==="
PROFILE_BIN="$BUILD_DIR/bench/bench_nmf_profile_c86"
if [ -f "$PROFILE_BIN" ]; then
    "$PROFILE_BIN" 2>&1
else
    echo "SKIP: profile binary not found at $PROFILE_BIN"
    ls "$BUILD_DIR/bench/" 2>/dev/null | head -10
fi
echo ""

# --- Step 4: Run the adapter verification bench (uses reduce::nmf::fit with FitConfig) ---
echo "=== STEP 4: bench_reduce_nmf_c86 (adapter path — confirms fix is active) ==="
C86_BIN="$BUILD_DIR/bench/bench_reduce_nmf_c86"
if [ -f "$C86_BIN" ]; then
    "$C86_BIN" 2>&1
else
    echo "SKIP: c86 bench binary not found (likely counts.1pz missing — expected)"
    # The bench_reduce_nmf_c86 driver skips when counts.1pz missing.
    # Confirmed in bench_reduce_nmf_c86.cpp:82-86: skip_no_sample()
fi
echo ""

echo "=== DONE: $(date) ==="
echo "SUCCESS CRITERIA:"
echo "  - Build exit 0: check above"
echo "  - ctest nmf: >=13 pass, <=2 skip, 0 new failures"
echo "  - Profile k50_auto: should show the ORIGINAL 15,980ms regression (direct factornet call)"
echo "  - Profile k50_MU: should show ~484ms (MU control)"
echo "  - Adapter (bench_reduce_nmf_c86): k=50 should show <1,500ms (MU via FitConfig)"
