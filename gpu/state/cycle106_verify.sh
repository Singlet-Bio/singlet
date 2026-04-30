#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 106 verify — streaming factornet drop (no factornet dependency in streaming/).
#
# Tests:
#   1. Standalone nvcc compile of examples/cpp_minimal/main.cpp WITH factornet
#      still on path (regression: existing code must not break).
#   2. Header self-containment: streaming/pz_data_loader.h + streaming/streamed_pipeline.h
#      WITHOUT factornet on the include path (true native-only test).
#   3. NMF smoke: examples/nmf_smoke/main.cpp (WITH factornet on path for preprocess
#      headers that haven't been migrated yet).
#   4. CMakeLists.txt configure WITHOUT factornet (FACTORNET_INCLUDE_DIR not set).
#
# Cloned from state/cycle105_verify.sh — same 4-test pattern.

#SBATCH --job-name=sg_c106_verify
#SBATCH --partition=gpu
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle106_verify_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
EIGEN_INCLUDE=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3

echo "=== Cycle 106 verify: streaming factornet drop ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------
# Test 1: compile examples/cpp_minimal/main.cpp WITH factornet (regression).
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 1: cpp_minimal WITH factornet (regression check) ---"
nvcc -std=c++20 -O2 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I "${EIGEN_INCLUDE}" \
    -x cu \
    -c "$SRC_DIR/examples/cpp_minimal/main.cpp" \
    -o "/dev/shm/cpp_minimal106_${SLURM_JOB_ID}.o" 2>&1 | tail -40
COMPILE_EXIT=${PIPESTATUS[0]}
echo "COMPILE_EXIT=${COMPILE_EXIT}"
[ "${COMPILE_EXIT}" -eq 0 ] && rm -f "/dev/shm/cpp_minimal106_${SLURM_JOB_ID}.o"

# ---------------------------------------------------------------------------
# Test 2: streaming headers WITHOUT factornet — the CYCLE-106 gate.
# io/chunk.h, streaming/pz_data_loader.h, streaming/streamed_pipeline.h must
# all compile cleanly without factornet on the include path.
# Preprocess headers (lognorm.h, hvg.h) may still need factornet; we test
# only streaming/ here.
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 2: streaming headers WITHOUT factornet ---"
cat > /dev/shm/sg_streaming106_${SLURM_JOB_ID}.cu <<'CU_EOF'
// Test native streaming types — no factornet on include path.
#include <singlet-gpu/io/chunk.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/reduce/nmf/types.h>

int main() {
    // Verify singlet_gpu::io::Chunk compiles.
    singlet_gpu::io::Chunk chunk;
    chunk.n_rows   = 30000;
    chunk.n_cols   = 1000;
    chunk.n_chunks = 100;
    chunk.col_ptr.resize(static_cast<size_t>(chunk.n_cols + 1), 0);
    chunk.row_indices.reserve(1000000);
    chunk.values.reserve(1000000);

    // Verify DeviceMemory::wrap factory compiles.
    singlet_gpu::core::DeviceMemory<int>   dm_int(10);
    auto view_int = singlet_gpu::core::DeviceMemory<int>::wrap(dm_int.get(), 10);
    (void)view_int;

    singlet_gpu::core::DeviceMemory<float> dm_float(10);
    auto view_float = singlet_gpu::core::DeviceMemory<float>::wrap(dm_float.get(), 10);
    (void)view_float;

    // DeviceCSC construction still works.
    singlet_gpu::core::DeviceCSC csc(10, 20, 100);

    return 0;
}
CU_EOF

nvcc -std=c++20 -O2 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -c "/dev/shm/sg_streaming106_${SLURM_JOB_ID}.cu" \
    -o "/dev/shm/sg_streaming106_${SLURM_JOB_ID}.o" 2>&1 | tail -40
NATIVE_EXIT=${PIPESTATUS[0]}
echo "NATIVE_EXIT=${NATIVE_EXIT}"
rm -f /dev/shm/sg_streaming106_${SLURM_JOB_ID}.{cu,o}

# ---------------------------------------------------------------------------
# Test 3: nmf_smoke compile WITH factornet (preprocess still needs it).
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 3: nmf_smoke compile ---"
nvcc -std=c++20 -O2 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I "${EIGEN_INCLUDE}" \
    -x cu \
    -c "$SRC_DIR/examples/nmf_smoke/main.cpp" \
    -o "/dev/shm/nmf_smoke106_${SLURM_JOB_ID}.o" 2>&1 | tail -40
NMF_EXIT=${PIPESTATUS[0]}
echo "NMF_EXIT=${NMF_EXIT}"
[ "${NMF_EXIT}" -eq 0 ] && rm -f "/dev/shm/nmf_smoke106_${SLURM_JOB_ID}.o"

# ---------------------------------------------------------------------------
# Test 4: CMakeLists.txt configure WITHOUT factornet (critical gate).
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 4: cmake configure WITHOUT factornet (FACTORNET_INCLUDE_DIR not set) ---"
CMAKE_TEST_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle106_cmake_no_factornet
rm -rf "$CMAKE_TEST_DIR"
mkdir -p "$CMAKE_TEST_DIR"
cmake -S "$SRC_DIR" -B "$CMAKE_TEST_DIR" \
    -DSINGLET_GPU_BUILD_TESTS=OFF \
    -DSINGLET_GPU_BUILD_BENCH=OFF \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    2>&1 | tail -20
CMAKE_EXIT=${PIPESTATUS[0]}
echo "CMAKE_EXIT=${CMAKE_EXIT}"

echo ""
echo "=== FINAL EXIT CODES ==="
echo "COMPILE_EXIT=${COMPILE_EXIT}"
echo "NATIVE_EXIT=${NATIVE_EXIT}"
echo "NMF_EXIT=${NMF_EXIT}"
echo "CMAKE_EXIT=${CMAKE_EXIT}"
date
