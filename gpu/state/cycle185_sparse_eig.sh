#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-185 iter-3: re-verify core/sparse_eigensolver after relative-residual fix.
# Root cause: absolute convergence max|R| < tol stalls when spectral_radius * eps_fp32 > tol.
# Fix: lobpcg_abs_kernel now computes |R[i,k]|/|rho[k]| → scale-independent relative criterion.
# CYCLE-183 iter-2 (job 372480): 2/5 PASS — convergence check broken for non-diagonal A.

#SBATCH --job-name=sparse_eig_c185
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle185_sparse_eig_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle185_sparse_eig
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== CYCLE-185: core/sparse_eigensolver correctness verify (relative-residual fix) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF 2>&1 | tail -20
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"

echo ""
echo "--- cmake build: core_sparse_eigensolver_correctness ---"
cmake --build "$BUILD_DIR" \
    --target core_sparse_eigensolver_correctness \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"

if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — stopping."
    exit 1
fi

TEST_BIN="$BUILD_DIR/tests/core_sparse_eigensolver_correctness"

if [ ! -x "$TEST_BIN" ]; then
    echo "TEST BINARY NOT FOUND at $TEST_BIN — stopping."
    exit 1
fi

# Copy scipy reference into the expected /tmp path on this node.
echo ""
echo "--- copy scipy reference to compute node /tmp ---"
mkdir -p /tmp/singlet_gpu_sparse_eig_refs_tmp
if [ -f /mnt/home/debruinz/Singlet-AI/singlet-gpu/tests/refs/sparse_eig_medium_ref.npz ]; then
    cp /mnt/home/debruinz/Singlet-AI/singlet-gpu/tests/refs/sparse_eig_medium_ref.npz \
       /tmp/singlet_gpu_sparse_eig_refs_tmp/sparse_eig_medium_ref.npz
elif [ -f /tmp/singlet_gpu_sparse_eig_refs_tmp/sparse_eig_medium_ref.npz ]; then
    echo "ref already in /tmp on this node"
else
    echo "WARNING: scipy reference npz not found; Test 2 will SKIP"
fi
ls -la /tmp/singlet_gpu_sparse_eig_refs_tmp/ 2>/dev/null

# Run the 5 tests, one filter at a time.
declare -a TESTS=(
    "TinyEigSymmetric"
    "ScaleSmokeMedium"
    "Determinism_BitIdentical"
    "Convergence_OnConvergent"
    "KneeOversampleRobustness"
)

declare -a EXITS=()

for TEST_NAME in "${TESTS[@]}"; do
    echo ""
    echo "--- $TEST_NAME ---"
    "$TEST_BIN" --gtest_color=no --gtest_filter="*.${TEST_NAME}" 2>&1
    EXITS+=("$?")
    echo "exit=${EXITS[-1]}"
done

echo ""
echo "=== SUMMARY ==="
echo "Build:                                $([ $BUILD_EXIT -eq 0 ] && echo PASS || echo FAIL)"
for i in "${!TESTS[@]}"; do
    printf "Test %d %-30s %s\n" "$((i+1))" "${TESTS[i]}:" "$([ ${EXITS[i]} -eq 0 ] && echo PASS || echo FAIL)"
done

OVERALL=0
for EXIT in "${EXITS[@]}"; do
    [ "$EXIT" -ne 0 ] && OVERALL=1
done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
