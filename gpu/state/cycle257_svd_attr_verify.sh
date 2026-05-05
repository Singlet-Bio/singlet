#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-257: verify the SvdResult.rows/cols Python-side workaround
# (CYCLE-199 root cause was AttributeError, NOT a kernel segfault — the
# CYCLE-255 deflation status-check audit unmasked it).
#
# Fix in python/singlet_gpu/reduce/svd.py:115-128: replaced
# result.rows/result.cols (not exposed via pybind11) with
# adata.n_vars/adata.n_obs (matches the wrapper's pre-SVD axis swap at
# line 97).
#
# Verify: rerun the 7 test files that were BLOCKED on CYCLE-199:
#   test_pp_neighbors, test_reduce, test_integrate,
#   test_tl_leiden, test_tl_rank_genes_groups, test_tl_umap, test_velocity
# Expected: ≥ 1 of these flips PASS at file level. Even partial PASS is
# enough to confirm CYCLE-199 is NOT a kernel bug; full PASS means the
# 7-file BLOCKED group is closed.

#SBATCH --job-name=sg_c257_svd_attr_verify
#SBATCH --partition=gpu
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle257_svd_attr_verify_%j.log

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
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle257_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python
VENV=/tmp/singlet_gpu_venv_${SLURM_JOB_ID}

echo "=== CYCLE-257: SVD attribute fix verify (CYCLE-199 7-file unblock) ==="
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

for tf in test_pp_neighbors.py test_reduce.py test_integrate.py \
          test_tl_leiden.py test_tl_rank_genes_groups.py \
          test_tl_umap.py test_velocity.py test_de_pseudobulk.py; do
  echo ""
  echo "================================================================"
  echo "--- pytest $tf ---"
  echo "================================================================"
  timeout 600 python -m pytest "$tf" -v --tb=line 2>&1 | tail -25 || true
done

echo ""
echo "================================================================"
echo "--- SUMMARY ---"
echo "================================================================"
echo "All 7 CYCLE-199 test files + test_de_pseudobulk run above."
echo "If most PASS or skip cleanly: CYCLE-199 BLOCKED can be closed."

deactivate
rm -rf "$VENV"
date
exit 0
