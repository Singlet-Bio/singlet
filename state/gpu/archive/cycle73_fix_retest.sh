#!/bin/bash
#SBATCH --job-name=sg_cycle73_fix
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle73_fix_retest_%j.log

set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle73_verify
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 73 fix retest ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Build dir: $BUILD_DIR"
date

# Fresh build directory
rm -rf "$BUILD_DIR"

echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1
CMAKE_EXIT=$?
echo "CMake exit: $CMAKE_EXIT"

if [ "$CMAKE_EXIT" -ne 0 ]; then
    echo "CMAKE CONFIGURE FAILED — aborting"
    exit $CMAKE_EXIT
fi

echo "--- cmake build (target de_wilcoxon_correctness) ---"
cmake --build "$BUILD_DIR" --target de_wilcoxon_correctness -j8 2>&1
BUILD_EXIT=$?
echo "Build exit: $BUILD_EXIT"

if [ "$BUILD_EXIT" -ne 0 ]; then
    echo "BUILD FAILED — aborting tests"
    exit $BUILD_EXIT
fi

echo "--- ctest: Test83_RealDataSized_PostNormalize_NoCrash ---"
set +e
ctest --test-dir "$BUILD_DIR" \
    -R "Test83_RealDataSized_PostNormalize_NoCrash" \
    --output-on-failure \
    -V \
    2>&1
TEST83_EXIT=$?
echo "Test83 exit: $TEST83_EXIT"

echo "--- ctest: Wilcoxon.*TinyPlanted (no-crash sanity) ---"
ctest --test-dir "$BUILD_DIR" \
    -R "Wilcoxon.*TinyPlanted" \
    --output-on-failure \
    -V \
    2>&1
TINY_EXIT=$?
echo "TinyPlanted exit: $TINY_EXIT"
set -e

echo "=== Done ==="
echo "Build exit=$BUILD_EXIT  Test83 exit=$TEST83_EXIT  TinyPlanted exit=$TINY_EXIT"
date
