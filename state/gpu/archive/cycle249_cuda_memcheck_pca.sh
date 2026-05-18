#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-249: cuda-memcheck (compute-sanitizer) wrapper for the
# pca C++ kernel segfault (CYCLE-199).
#
# After CYCLE-215 minimal repro localized the bug to scale ≥ 5000×2000 (the kernel HAS
# isfinite() guards + in-bounds indexing), the only path forward is
# actual GPU memory-checker output that localizes the offending kernel.
#
# This script:
#   1. Builds the wheel with the pinned-deps test environment.
#   2. Re-runs ONLY test_pseudobulk_de_min_cells_filter under
#      compute-sanitizer (the cuda-memcheck successor in CUDA 12+).
#   3. Captures the offending instruction + kernel name.
#
# Output goes to state/cycle249_pca_memcheck_${SLURM_JOB_ID}.log.
# Look for lines like:
#   ========= Invalid __global__ read of size N
#   =========     at kernel_name<...> in /path/to/header.h:LINE
#
# Cost: ~10-15 min (sanitizer adds 5-10× slowdown but the test is small).

#SBATCH --job-name=sg_c249_pca_memcheck
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle249_pca_memcheck_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc
export CMAKE_PREFIX_PATH="/opt/rh/gcc-toolset-13/root/usr/lib"

EIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3
export SKBUILD_CMAKE_ARGS="-DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR};-DEigen3_DIR=${EIGEN_INCLUDE_DIR}/cmake"
export CMAKE_ARGS="-DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}"

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>&1 || echo "WARN: module load failed"

export PIP_CACHE_DIR=/mnt/projects/debruinz_project/singlet-gpu/.pip-cache
mkdir -p "$PIP_CACHE_DIR"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle249_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python
VENV=/tmp/singlet_gpu_venv_${SLURM_JOB_ID}

echo "=== CYCLE-244: pseudobulk min_cells_filter compute-sanitizer ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
which compute-sanitizer 2>&1 || echo "WARN: compute-sanitizer not in PATH"

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --user --upgrade 2>&1 | tail -5
fi

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet
python -m pip install \
    "scikit-build-core>=0.9" "pybind11>=2.11" "pytest>=7.0" \
    "numpy>=1.24,<2.6" "scipy>=1.11,<1.18" "anndata>=0.10,<0.13" \
    "scanpy>=1.10,<1.12" "cupy-cuda12x>=13.0,<15" 2>&1 | tail -10
python -m pip install -e "$PY_DIR" --no-build-isolation 2>&1 | tail -5

echo ""
echo "--- compute-sanitizer pytest test_pseudobulk_de_min_cells_filter ---"
# compute-sanitizer flags:
#   --tool memcheck       — detect OOB / unaligned / leaked allocations
#   --launch-timeout 0    — allow long-running pytest setup
#   --target-processes all — needed because pytest spawns workers
#   --print-limit 4       — only print first 4 errors per kernel (full trace
#                            on first error is enough for diagnosis)
# Need to cd into the tests dir so pytest finds conftest.py.
cd "$PY_DIR/tests"

compute-sanitizer \
    --tool memcheck \
    --launch-timeout 0 \
    --target-processes all \
    --print-limit 4 \
    --log-file=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle249_sanitizer_${SLURM_JOB_ID}.log \
    python -m pytest \
        "test_pp_neighbors.py::test_neighbors_basic" \
        --runxfail \
        -v --tb=short 2>&1 | tail -60

echo ""
echo "--- compute-sanitizer log: ---"
cat /mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle249_sanitizer_${SLURM_JOB_ID}.log 2>&1 | head -80

deactivate
rm -rf "$VENV"
date
exit 0
