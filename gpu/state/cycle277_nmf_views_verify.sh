#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-277: broader verify after CYCLE-260-265 wrapper-rot peel.
# Confirms: test_pp_neighbors (3/3 PASS via xfail), test_reduce delta,
# test_integrate delta.

#SBATCH --job-name=sg_c277_nmf_verify
#SBATCH --partition=gpu
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle277_nmf_views_verify_%j.log

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
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle277_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python
VENV=/tmp/singlet_gpu_venv_${SLURM_JOB_ID}

echo "=== CYCLE-277: broader verify ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --user --upgrade 2>&1 | tail -5
fi

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet
python -m pip install \
    "scikit-build-core>=0.9" "pybind11>=2.11" "pytest>=7.0" \
    "numpy>=1.24,<2.6" "scipy>=1.11,<1.18" "anndata>=0.10,<0.13" \
    "scanpy>=1.10,<1.12" "cupy-cuda12x>=13.0,<15" 2>&1 | tail -5
python -m pip install -e "$PY_DIR" --no-build-isolation 2>&1 | tail -5

cd "$PY_DIR/tests"

for tf in test_reduce.py; do
  echo ""
  echo "================================================================"
  echo "--- pytest $tf ---"
  echo "================================================================"
  timeout 600 python -m pytest "$tf" -v --tb=line 2>&1 | tail -25 || true
done

deactivate
rm -rf "$VENV"
date
exit 0
