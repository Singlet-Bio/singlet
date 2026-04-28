#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/cycle57_bench_job.sh
#
# Cycle 57 — Feature 0 Phase E benchmark job script.
# Run on g001 (Tesla V100S sm_70) ONLY.
#
# Submit: sbatch bench/cycle57_bench_job.sh
#   OR interactive: salloc -w g001 -p gpu --gres=gpu:1 --cpus-per-task=16 \
#                          --mem=64G --time=2:00:00
#                   then: bash bench/cycle57_bench_job.sh
#
#SBATCH --job-name=singlet-gpu-cycle57-bench
#SBATCH --partition=gpu
#SBATCH --nodelist=g001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle57_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle57_%j.err

set -euo pipefail

SINGLET_ROOT="/mnt/home/debruinz/Singlet-AI/singlet-gpu"
FACTORNET_ROOT="/mnt/home/debruinz/factornet"
BUILD_DIR="${SINGLET_ROOT}/build"
LOG_DIR="${SINGLET_ROOT}/bench/logs"
REFS_DIR="${SINGLET_ROOT}/bench/refs"

mkdir -p "${LOG_DIR}"
echo "=== Cycle 57 Phase E Benchmark ==="
echo "Node: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "CUDA: $(nvcc --version 2>/dev/null | tail -1 || echo 'nvcc not on PATH')"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Step 0: Activate gcc-toolset-13 + CUDA PATH (required on g001).
# ---------------------------------------------------------------------------
source /opt/rh/gcc-toolset-13/enable 2>/dev/null || true
# Set CUDA path — /usr/local/cuda symlink points to current CUDA on Clipper.
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDACXX="/usr/local/cuda/bin/nvcc"
echo "GCC: $(gcc --version | head -1)"
echo "NVCC: $(nvcc --version 2>/dev/null | tail -1)"
echo "CUDACXX: ${CUDACXX}"

# ---------------------------------------------------------------------------
# Step 1: Clean build.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Clean build ---"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Eigen3 on Clipper lives in a spack-managed location (not /usr/include/eigen3).
EIGEN3_SPACK="/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3"
cmake "${SINGLET_ROOT}" \
    -DFACTORNET_ROOT="${FACTORNET_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DEIGEN_INCLUDE_DIR="${EIGEN3_SPACK}" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=ON \
    2>&1 | tail -30

echo ""
echo "--- Building io_pz_device_loader_correctness (test) ---"
make -j16 io_pz_device_loader_correctness 2>&1 | grep -E 'error:|warning:|Built|Linking' | head -40

echo ""
echo "--- Building io_pz_device_loader_bench (Cycle 57) ---"
make -j16 io_pz_device_loader_bench 2>&1 | grep -E 'error:|warning:|Built|Linking' | head -40

# ---------------------------------------------------------------------------
# Step 2: Verify Cycle 56 correctness tests still pass.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Cycle 56 correctness re-verification ---"
if [ -f "${BUILD_DIR}/tests/io_pz_device_loader_correctness" ]; then
    "${BUILD_DIR}/tests/io_pz_device_loader_correctness" \
        --gtest_filter="*RealSample*:*AutoTune*:*ChunkIterator*:*MatrixType*" \
        2>&1 | tail -20
    echo "Correctness re-verify exit code: $?"
else
    echo "WARNING: correctness binary not found at ${BUILD_DIR}/tests/io_pz_device_loader_correctness"
    echo "         Continuing benchmark — correctness check skipped."
fi

# ---------------------------------------------------------------------------
# Step 3: Set up Python environment.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Python environment check ---"
PYTHON3="python3"
${PYTHON3} --version
${PYTHON3} -c "import scanpy; print('scanpy', scanpy.__version__)" 2>/dev/null || echo "  scanpy: not available"
${PYTHON3} -c "import anndata; print('anndata', anndata.__version__)" 2>/dev/null || echo "  anndata: not available"
${PYTHON3} -c "import cupy; print('cupy', cupy.__version__)" 2>/dev/null || echo "  cupy: not available"
${PYTHON3} -c "import h5py; print('h5py', h5py.__version__)" 2>/dev/null || echo "  h5py: not available"
${PYTHON3} -c "import scipy; print('scipy', scipy.__version__)" 2>/dev/null || echo "  scipy: not available"

# ---------------------------------------------------------------------------
# Step 4: Run format conversion (one-time, not timed).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Baseline format conversion ---"
PZ_SMALL="/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz"
H5_SMALL="/dev/shm/singlet_gpu_bench_small.h5"
H5AD_SMALL="/dev/shm/singlet_gpu_bench_small.h5ad"

if [ -f "${PZ_SMALL}" ]; then
    ${PYTHON3} "${REFS_DIR}/cycle57_convert.py" \
        --pz="${PZ_SMALL}" \
        --h5="${H5_SMALL}" \
        --h5ad="${H5AD_SMALL}" \
        2>&1 || echo "  WARNING: conversion failed (baselines will be skipped)"
else
    echo "  WARNING: ${PZ_SMALL} not found — small scale baselines will be skipped"
fi

# ---------------------------------------------------------------------------
# Step 5: Run Nsys profile for our_manual on small scale (non-blocking).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: Nsys profile (our_manual, small scale) ---"
TRACES_DIR="${SINGLET_ROOT}/bench/traces"
mkdir -p "${TRACES_DIR}"
if command -v nsys &>/dev/null && [ -f "${BUILD_DIR}/bench/io_pz_device_loader_bench" ]; then
    echo "  Running nsys profile ..."
    nsys profile \
        --output="${TRACES_DIR}/cycle57_small_our_manual" \
        --trace=cuda,nvtx,osrt \
        --force-overwrite=true \
        --stats=true \
        --duration=60 \
        "${BUILD_DIR}/bench/io_pz_device_loader_bench" \
        2>&1 | head -60 || echo "  nsys profile failed (non-fatal)"
else
    echo "  nsys not available or bench binary missing — skipping profile"
fi

# ---------------------------------------------------------------------------
# Step 6: Run the full benchmark.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: Run benchmark ---"
if [ ! -f "${BUILD_DIR}/bench/io_pz_device_loader_bench" ]; then
    echo "ERROR: benchmark binary not found — build failed."
    exit 1
fi

"${BUILD_DIR}/bench/io_pz_device_loader_bench" 2>&1
BENCH_EXIT=$?

echo ""
echo "Benchmark exit code: ${BENCH_EXIT}"
if [ "${BENCH_EXIT}" -eq 0 ]; then
    echo "RESULT: FRONTIER PROMOTED"
elif [ "${BENCH_EXIT}" -eq 2 ]; then
    echo "RESULT: NOT PROMOTED (needs Phase D iteration)"
else
    echo "RESULT: BENCH FAILED"
fi

# ---------------------------------------------------------------------------
# Step 7: Show updated benchmark-registry.md tail.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 7: benchmark-registry.md (last 20 rows) ---"
tail -20 "${SINGLET_ROOT}/state/benchmark-registry.md" || true

echo ""
echo "=== Cycle 57 complete. ==="
echo "Log: ${LOG_DIR}/cycle57_${SLURM_JOB_ID:-local}.log"
