#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 104 pybind verify — full _core.so build via python -m pip install -e python/
# Validates CYCLE-103 wrappers + CYCLE-104 unblock fixes end-to-end.

#SBATCH --job-name=sg_c104_pip
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle104_pip_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc
export CMAKE_PREFIX_PATH="/opt/rh/gcc-toolset-13/root/usr/lib"

# Eigen 3.4.0 is required transitively by factornet. The Clipper Spack tree has
# it under /opt/gvsu/clipper/2024.05/...  Pass via SKBUILD_CMAKE_ARGS so it
# reaches scikit-build-core's cmake configure without baking the path into
# pyproject.toml (which would break wheels on other systems).
EIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3
export SKBUILD_CMAKE_ARGS="-DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR};-DEigen3_DIR=${EIGEN_INCLUDE_DIR}/cmake"
# Some scripts use CMAKE_ARGS directly (older scikit-build versions). Set both.
export CMAKE_ARGS="-DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}"

# pyproject.toml requires Python >= 3.10; system python is 3.9.23. Load 3.11
# from the cluster module system. Module names from `module avail python` on
# Clipper: python/3.10.19, 3.11.14, 3.12.12, 3.13.8, 3.14.0 (default 3.9.23).
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>&1 || echo "WARN: module load failed; falling back to system python"

# Outputs go under projects (home is at quota)
export PIP_CACHE_DIR=/mnt/projects/debruinz_project/singlet-gpu/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

# Build artifacts also under projects
export CMAKE_BUILD_PARALLEL_LEVEL=8
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle104_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python

echo "=== Cycle 104 pybind verify ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "Python: $(python --version)  (which: $(which python))"

# The Spack-built Python module ships without pip. Bootstrap if missing.
if ! python -m pip --version >/dev/null 2>&1; then
    echo "pip not available — bootstrapping via ensurepip..."
    python -m ensurepip --user --upgrade 2>&1 | tail -5
    # ensurepip --user adds ~/.local/bin to PATH only on first login;
    # use python -m pip which doesn't need the script symlink.
fi
echo "pip:    $(python -m pip --version 2>&1)"

rm -rf "$SKBUILD_BUILD_DIR"

echo ""
echo "--- python -m pip install -e python/ ---"
cd "$PY_DIR"
python -m pip install --user -e . --no-build-isolation 2>&1 | tail -80
PIP_EXIT=${PIPESTATUS[0]}
echo "PIP_EXIT=${PIP_EXIT}"

if [ "${PIP_EXIT}" -ne 0 ]; then
    echo "=== python -m pip install FAILED ==="
    # Try with build isolation in case of missing build deps
    echo "Retrying with build isolation..."
    python -m pip install --user -e . 2>&1 | tail -80
    PIP_EXIT2=${PIPESTATUS[0]}
    echo "PIP_EXIT2=${PIP_EXIT2}"
    if [ "${PIP_EXIT2}" -ne 0 ]; then
        echo "=== python -m pip install FAILED (both attempts) ==="
        exit "${PIP_EXIT2}"
    fi
fi

echo ""
echo "--- import smoke test ---"
python -c "
import singlet_gpu
print(f'singlet_gpu version: {singlet_gpu.__version__}')
print()
# CYCLE-103 functions:
from singlet_gpu.qc import calculate_qc_metrics, filter_cells, filter_genes
print('  qc.calculate_qc_metrics:', callable(calculate_qc_metrics))
print('  qc.filter_cells:        ', callable(filter_cells))
print('  qc.filter_genes:        ', callable(filter_genes))

from singlet_gpu.preprocess import scale, regress_out
print('  preprocess.scale:       ', callable(scale))
print('  preprocess.regress_out: ', callable(regress_out))

# Existing wrappers still work:
from singlet_gpu import load_pz
print('  load_pz:                ', callable(load_pz))
" 2>&1
IMPORT_EXIT=${PIPESTATUS[0]}
echo "IMPORT_EXIT=${IMPORT_EXIT}"

echo ""
echo "=== FINAL EXIT CODES ==="
echo "PIP_EXIT=${PIP_EXIT}"
echo "IMPORT_EXIT=${IMPORT_EXIT}"
date
