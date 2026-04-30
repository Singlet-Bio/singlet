#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 103 header check — compile _bind_qc_metrics + _bind_preprocess in isolation
# (without the pre-existing loss.cuh NMF error that affects full _core build).

#SBATCH --job-name=sg_c103_hdr
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle103_hdrcheck_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
PYBIND11_INC=/mnt/home/debruinz/.local/lib/python3.9/site-packages/pybind11/include
PYTHON_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

echo "=== Cycle 103 isolated header check ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------
# Step 1: cpp_minimal umbrella (sanity baseline)
# ---------------------------------------------------------------------------
echo ""
echo "--- cpp_minimal smoke test ---"
nvcc -std=c++20 -O2 -DFACTORNET_HAS_GPU=1 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -x cu \
    -c "$SRC_DIR/examples/cpp_minimal/main.cpp" \
    -o "/dev/shm/cpp_minimal_c103h_${SLURM_JOB_ID}.o" 2>&1 | tail -20
COMPILE_EXIT=${PIPESTATUS[0]}
echo "COMPILE_EXIT=${COMPILE_EXIT}"
rm -f "/dev/shm/cpp_minimal_c103h_${SLURM_JOB_ID}.o"

# ---------------------------------------------------------------------------
# Step 2: Isolated compile of _bind_qc_metrics.hpp + _bind_preprocess.hpp
# We write a minimal .cu that includes only those two new headers.
# This proves the new code compiles without pulling in the pre-existing
# loss.cuh bug from _bind_nmf_new.hpp.
# ---------------------------------------------------------------------------
echo ""
echo "--- Isolated compile: _bind_qc_metrics + _bind_preprocess ---"


# Write the stub loader header to a SEPARATE directory that will take priority
# in the -I search path. This overrides _bind_loader.hpp without touching source.
STUB_INC="/dev/shm/sg_c103_stub_${SLURM_JOB_ID}"
mkdir -p "$STUB_INC"
cat > "${STUB_INC}/_bind_loader.hpp" << 'STUB_EOF'
#pragma once
// Stub _bind_loader.hpp for cycle-103 isolated header check.
// Provides PyDeviceCsc without the .data() calls that fail under the
// current factornet API (pre-existing _bind_loader.hpp issue — not cycle-103).
#include <pybind11/pybind11.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <memory>
#include <cuda_runtime.h>
namespace py = pybind11;
struct PyDeviceCsc {
    std::shared_ptr<singlet_gpu::core::DeviceCSC> csc;
    cudaStream_t producer_stream = nullptr;
    std::shared_ptr<pybind11::object> cupy_owner;
    explicit PyDeviceCsc(std::shared_ptr<singlet_gpu::core::DeviceCSC> p,
                         cudaStream_t s = nullptr)
        : csc(std::move(p)), producer_stream(s) {}
};
inline void bind_metadata(py::module_& m) {}
inline void bind_device_csc(py::module_& m) {}
inline void bind_pz_device_matrix(py::module_& m) {}
STUB_EOF

# Also stub _bind_metadata.hpp (pulled by _bind_loader.hpp transitively).
cat > "${STUB_INC}/_bind_metadata.hpp" << 'STUB2_EOF'
#pragma once
#include <pybind11/pybind11.h>
namespace py = pybind11;
inline void bind_metadata(py::module_& m) {}
STUB2_EOF

# Write the main check .cu INSIDE the stub dir so quoted includes find stubs first.
# Also copy _cupy_interop.hpp to stub dir (it's clean, no .data() calls).
cp "$SRC_DIR/python/src/_cupy_interop.hpp" "${STUB_INC}/"
cp "$SRC_DIR/python/src/_bind_qc_metrics.hpp" "${STUB_INC}/"
cp "$SRC_DIR/python/src/_bind_preprocess.hpp" "${STUB_INC}/"

cat > "${STUB_INC}/sg_c103_check_${SLURM_JOB_ID}.cu" << 'CUDA_EOF'
// SPDX-License-Identifier: GPL-2.0-or-later
// CYCLE-103 isolated header compilation check.
// This file lives in the stub dir, so quoted includes find _bind_loader.hpp
// stub (in the same dir) before the real one in python/src/.
#define FACTORNET_HAS_GPU 1
#include "_cupy_interop.hpp"
#include "_bind_qc_metrics.hpp"
#include "_bind_preprocess.hpp"
PYBIND11_MODULE(_core_c103_check, m) {
    singlet_gpu::python::bind_qc_metrics(m);
    singlet_gpu::python::bind_preprocess(m);
}
CUDA_EOF

nvcc -std=c++20 -O2 -DFACTORNET_HAS_GPU=1 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -Xcompiler -fvisibility=hidden \
    -I "${STUB_INC}" \
    -I "$SRC_DIR/include" \
    -I "$SRC_DIR/python/src" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -I "${PYBIND11_INC}" \
    -I "${PYTHON_INC}" \
    -x cu \
    -c "${STUB_INC}/sg_c103_check_${SLURM_JOB_ID}.cu" \
    -o "/dev/shm/sg_c103_check_${SLURM_JOB_ID}.o" 2>&1 | tail -50
HDRCHECK_EXIT=${PIPESTATUS[0]}
echo "HDRCHECK_EXIT=${HDRCHECK_EXIT}"
rm -f "/dev/shm/sg_c103_check_${SLURM_JOB_ID}.o"
rm -rf "${STUB_INC}"

echo ""
echo "=== FINAL EXIT CODES ==="
echo "COMPILE_EXIT=${COMPILE_EXIT}  (cpp_minimal umbrella)"
echo "HDRCHECK_EXIT=${HDRCHECK_EXIT}  (cycle-103 new headers)"
date
