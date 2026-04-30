#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 88 verify — build + ctest scry deviance HVG
# Replaces cycle88_build.sh (job 367571) which failed because nvcc was not on PATH.

#SBATCH --job-name=sg_c88_verify
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle88_verify_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 88 verify: scry deviance HVG ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

# Fresh build dir to avoid cmake cache contamination from prior failed attempt
rm -rf "$BUILD_DIR"

echo ""
echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1 | tail -25
CMAKE_EXIT=${PIPESTATUS[0]}
echo "CMAKE_EXIT=${CMAKE_EXIT}"
[ "${CMAKE_EXIT}" -ne 0 ] && { echo "=== CMAKE CONFIGURE FAILED ==="; exit "${CMAKE_EXIT}"; }

echo ""
echo "--- cmake build: preprocess_hvg_deviance_correctness + preprocess_hvg_correctness ---"
cmake --build "$BUILD_DIR" \
    --target preprocess_hvg_deviance_correctness preprocess_hvg_correctness \
    -j8 2>&1 | tail -50
BUILD_EXIT=${PIPESTATUS[0]}
echo "BUILD_EXIT=${BUILD_EXIT}"
[ "${BUILD_EXIT}" -ne 0 ] && { echo "=== BUILD FAILED ==="; exit "${BUILD_EXIT}"; }

echo ""
echo "=== Direct test-binary execution (bypass ctest/gtest_discover_tests) ==="
echo "Note: ctest+gtest_discover_tests doesn't generate discovery files when"
echo "      cmake --build runs with --target X (only). Calling binaries directly."

DEV_BIN="$BUILD_DIR/tests/preprocess_hvg_deviance_correctness"
HVG_BIN="$BUILD_DIR/tests/preprocess_hvg_correctness"

DEVIANCE_EXIT=99
HVG_REG_EXIT=99

if [ -x "$DEV_BIN" ]; then
    echo ""
    echo "--- $DEV_BIN ---"
    "$DEV_BIN" --gtest_color=no 2>&1 | tail -60
    DEVIANCE_EXIT=${PIPESTATUS[0]}
fi
echo "DEVIANCE_EXIT=${DEVIANCE_EXIT}"

if [ -x "$HVG_BIN" ]; then
    echo ""
    echo "--- $HVG_BIN ---"
    "$HVG_BIN" --gtest_color=no 2>&1 | tail -60
    HVG_REG_EXIT=${PIPESTATUS[0]}
fi
echo "HVG_REG_EXIT=${HVG_REG_EXIT}"

echo ""
echo "=== FINAL EXIT CODES ==="
echo "CMAKE_EXIT=${CMAKE_EXIT}"
echo "BUILD_EXIT=${BUILD_EXIT}"
echo "DEVIANCE_EXIT=${DEVIANCE_EXIT}"
echo "HVG_REG_EXIT=${HVG_REG_EXIT}"
date
