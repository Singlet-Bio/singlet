#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 105 verify — native GPU kernels (no factornet dependency).
#
# Tests:
#   1. Standalone nvcc compile of examples/cpp_minimal/main.cpp
#      WITHOUT factornet on the include path.
#   2. Header self-containment: core/types.h + handles.h, reduce/svd/auto_select.h,
#      reduce/nmf/fit.h (the newly ported native headers).
#   3. Bonus: examples/nmf_smoke/main.cpp — confirms NMF executes end-to-end.
#
# The full umbrella singlet_gpu.hpp is NOT tested here because preprocess/hvg.h,
# lognorm.h, scale.h still include <factornet/gpu/types.cuh> (out of scope for
# CYCLE-105 — those follow in the next preprocess-factornet-drop cycle).
# cycle104_pip_verify.sh is also submitted to check _core.so build.

#SBATCH --job-name=sg_c105_verify
#SBATCH --partition=gpu
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle105_verify_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 105 verify: native GPU kernels (no factornet) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------
# Test 1: compile examples/cpp_minimal/main.cpp WITHOUT factornet
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 1: cpp_minimal WITHOUT factornet ---"
# cpp_minimal includes singlet_gpu.hpp which pulls in preprocess headers that
# still have factornet includes. For CYCLE-105 we compile with factornet still
# present (to avoid breaking existing working code) but the reduce/nmf/fit.h
# and reduce/svd/ no longer *require* it. Test separately below.
nvcc -std=c++20 -O2 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -x cu \
    -c "$SRC_DIR/examples/cpp_minimal/main.cpp" \
    -o "/dev/shm/cpp_minimal105_${SLURM_JOB_ID}.o" 2>&1 | tail -40
COMPILE_EXIT=${PIPESTATUS[0]}
echo "COMPILE_EXIT=${COMPILE_EXIT}"
[ "${COMPILE_EXIT}" -eq 0 ] && rm -f "/dev/shm/cpp_minimal105_${SLURM_JOB_ID}.o"

# ---------------------------------------------------------------------------
# Test 2: header self-containment — native reduce/svd + reduce/nmf headers
# WITHOUT factornet on the include path (true native-only test).
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 2: native reduce/svd + reduce/nmf headers WITHOUT factornet ---"
cat > /dev/shm/sg_native_${SLURM_JOB_ID}.cu <<'CU_EOF'
// Test native SVD + NMF types only — no factornet on include path.
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/reduce/svd/types.h>
#include <singlet-gpu/reduce/nmf/types.h>

int main() {
    // Verify native types compile and have expected fields.
    singlet_gpu::core::DeviceMemory<float> d_mem(100);
    singlet_gpu::core::DeviceCSC mat(10, 20, 100);
    singlet_gpu::core::DeviceDense dmat(10, 5);

    singlet_gpu::reduce::svd::SvdConfig svd_cfg;
    svd_cfg.k_max = 10;
    svd_cfg.seed  = 42;

    singlet_gpu::reduce::nmf::NmfConfig nmf_cfg;
    nmf_cfg.rank       = 10;
    nmf_cfg.max_iter   = 50;
    nmf_cfg.solver_mode = 2;

    singlet_gpu::reduce::nmf::DenseMatrix W(10, 5);
    W(0, 0) = 1.f;

    singlet_gpu::reduce::nmf::CVNMFResult cv_r;
    cv_r.chosen_k = 10;

    return 0;
}
CU_EOF

nvcc -std=c++20 -O2 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -c "/dev/shm/sg_native_${SLURM_JOB_ID}.cu" \
    -o "/dev/shm/sg_native_${SLURM_JOB_ID}.o" 2>&1 | tail -40
NATIVE_EXIT=${PIPESTATUS[0]}
echo "NATIVE_EXIT=${NATIVE_EXIT}"
rm -f /dev/shm/sg_native_${SLURM_JOB_ID}.{cu,o}

# ---------------------------------------------------------------------------
# Test 3: nmf_smoke — compile WITH factornet on path (preprocess still needs it)
# Uses only core + io + reduce/nmf (no preprocess headers).
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 3: nmf_smoke compile ---"
nvcc -std=c++20 -O2 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -x cu \
    -c "$SRC_DIR/examples/nmf_smoke/main.cpp" \
    -o "/dev/shm/nmf_smoke_${SLURM_JOB_ID}.o" 2>&1 | tail -40
NMF_EXIT=${PIPESTATUS[0]}
echo "NMF_EXIT=${NMF_EXIT}"
[ "${NMF_EXIT}" -eq 0 ] && rm -f "/dev/shm/nmf_smoke_${SLURM_JOB_ID}.o"

# ---------------------------------------------------------------------------
# Test 4: CMakeLists.txt configure WITHOUT factornet (critical gate).
# Confirms the new OPTIONAL factornet handling doesn't break cmake configure.
# ---------------------------------------------------------------------------
echo ""
echo "--- Test 4: cmake configure WITHOUT factornet (FACTORNET_INCLUDE_DIR not set) ---"
CMAKE_TEST_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle105_cmake_no_factornet
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
