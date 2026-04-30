#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 103 verify — compile cpp_minimal smoke test + _core pybind11 extension.

#SBATCH --job-name=sg_c103_verify
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle103_verify_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle103

echo "=== Cycle 103 verify: _bind_qc_metrics + _bind_preprocess ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------
# Step 1: cpp_minimal smoke test (umbrella header, no pybind11)
# ---------------------------------------------------------------------------
echo ""
echo "--- cpp_minimal smoke test ---"
nvcc -std=c++20 -O2 -DFACTORNET_HAS_GPU=1 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=sm_90 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -x cu \
    -c "$SRC_DIR/examples/cpp_minimal/main.cpp" \
    -o "/dev/shm/cpp_minimal_c103_${SLURM_JOB_ID}.o" 2>&1 | tail -40
COMPILE_EXIT=${PIPESTATUS[0]}
echo "COMPILE_EXIT=${COMPILE_EXIT}"
rm -f "/dev/shm/cpp_minimal_c103_${SLURM_JOB_ID}.o"

# ---------------------------------------------------------------------------
# Step 2: _core pybind11 extension via CMake
# ---------------------------------------------------------------------------
echo ""
echo "--- _core pybind11 extension build ---"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# WHY -S "$SRC_DIR" and not "$SRC_DIR/python":
#   python/CMakeLists.txt uses add_subdirectory("../") which loops when called
#   standalone (CMAKE_CURRENT_SOURCE_DIR == CMAKE_SOURCE_DIR). Instead we call
#   the parent CMakeLists.txt and add python/ as a subdirectory via a wrapper.
#   Simpler: use pip/scikit-build-core's intended entry point by passing the
#   parent SRC dir with SINGLET_GPU_BUILD_PYTHON=ON, but the parent CMakeLists
#   doesn't wire python/ in. So we use a temporary wrapper CMakeLists that just
#   calls add_subdirectory(python).

WRAPPER_DIR="/dev/shm/sg_c103_wrap_${SLURM_JOB_ID}"
mkdir -p "$WRAPPER_DIR"
cat > "$WRAPPER_DIR/CMakeLists.txt" <<'WRAP_EOF'
cmake_minimum_required(VERSION 3.24)
# Set SKBUILD_PROJECT_VERSION so python/CMakeLists.txt doesn't warn on VERSION.
set(SKBUILD_PROJECT_VERSION "0.1.0")
project(singlet_gpu_python_wrap LANGUAGES CXX CUDA)
add_subdirectory("@SRC_DIR@/python" python_build)
WRAP_EOF
# Substitute the actual path.
sed -i "s|@SRC_DIR@|${SRC_DIR}|g" "$WRAPPER_DIR/CMakeLists.txt"

PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || \
               echo "/mnt/home/debruinz/.local/lib/python3.9/site-packages/pybind11/share/cmake/pybind11")
PYTHON_EXEC=$(which python3)

cmake -S "$WRAPPER_DIR" -B "$BUILD_DIR" \
    -DSINGLET_GPU_BUILD_PYTHON=ON \
    -DSINGLET_GPU_BUILD_TESTS=OFF \
    -DSINGLET_GPU_BUILD_BENCH=OFF \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    -DPYTHON_EXECUTABLE="${PYTHON_EXEC}" \
    2>&1 | tail -30

CMAKE_CONF_EXIT=$?
echo "CMAKE_CONF_EXIT=${CMAKE_CONF_EXIT}"

if [ "${CMAKE_CONF_EXIT}" -eq 0 ]; then
    cmake --build "$BUILD_DIR" -j8 --target _core 2>&1 | tail -60
    PYBIND_EXIT=$?
else
    PYBIND_EXIT=99
fi
# Clean wrapper dir only after build (cmake re-checks source dir during build).
rm -rf "$WRAPPER_DIR"
echo "PYBIND_EXIT=${PYBIND_EXIT}"

# ---------------------------------------------------------------------------
# Step 3: Python import smoke test (does not require GPU at import time)
# ---------------------------------------------------------------------------
echo ""
echo "--- Python import smoke test ---"
CORE_SO=$(find "$BUILD_DIR" -name "_core*.so" 2>/dev/null | head -1)
if [ -n "${CORE_SO}" ] && [ "${PYBIND_EXIT}" -eq 0 ]; then
    PYTHONPATH="${BUILD_DIR}:${SRC_DIR}/python" \
    python3 -c "
import sys, os
# Insert the build dir so Python can find _core.so
sys.path.insert(0, os.path.dirname('${CORE_SO}'))
import singlet_gpu._core as _core
print('_core.__version__:', _core.__version__)
assert hasattr(_core, 'calculate_qc_metrics'), 'missing calculate_qc_metrics'
assert hasattr(_core, 'filter_cells'),         'missing filter_cells'
assert hasattr(_core, 'filter_genes'),         'missing filter_genes'
assert hasattr(_core, 'scale'),                'missing scale'
assert hasattr(_core, 'regress_out'),          'missing regress_out'
assert hasattr(_core, 'QcResult'),             'missing QcResult class'
assert hasattr(_core, 'DenseResult'),          'missing DenseResult class'
print('All 5 cycle-103 symbols present: PASS')
" 2>&1
    IMPORT_EXIT=${PIPESTATUS[0]}
else
    echo "Skipping import test (extension not built)"
    IMPORT_EXIT=99
fi
echo "IMPORT_EXIT=${IMPORT_EXIT}"

echo ""
echo "=== FINAL EXIT CODES ==="
echo "COMPILE_EXIT=${COMPILE_EXIT}"
echo "PYBIND_EXIT=${PYBIND_EXIT}"
echo "IMPORT_EXIT=${IMPORT_EXIT}"
date
