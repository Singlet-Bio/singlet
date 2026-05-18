#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-215: minimal pca repro to disambiguate CYCLE-199 root cause.
#
# Goal: call _core.pca on SYNTHETIC cupy sparse data (no lognorm chain, no
# adata path) at multiple sizes.  If pca segfaults on small synthetic
# matrices: it's a kernel bug.  If it only segfaults after our lognorm
# chain: it's a Python-side state/lifetime issue.

#SBATCH --job-name=sg_c215_pca_repro
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle215_pca_repro_%j.log

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
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle215_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python
VENV=/tmp/singlet_gpu_venv_${SLURM_JOB_ID}

echo "=== CYCLE-215: minimal pca repro ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --user --upgrade 2>&1 | tail -5
fi

python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet
python -m pip install \
    "scikit-build-core>=0.9" "pybind11>=2.11" \
    "numpy>=1.24,<2.6" "scipy>=1.11,<1.18" \
    "cupy-cuda12x>=13.0,<15" 2>&1 | tail -5
python -m pip install -e "$PY_DIR" --no-build-isolation 2>&1 | tail -5

echo ""
echo "--- pca minimal repro (no lognorm chain) ---"
python <<'PYEOF'
import sys
import numpy as np
import scipy.sparse as sp
import cupy as cp
try:
    import cupyx.scipy.sparse as csp
except ImportError:
    import cupy.sparse as csp

import singlet_gpu
import singlet_gpu._core as _core

print("singlet_gpu", singlet_gpu.__version__)
print("cupy", cp.__version__)

def try_pca(n_genes, n_cells, density=0.05, k=10, label=""):
    """Build synthetic genes×cells sparse, upload, call _core.pca, report."""
    np.random.seed(0)
    rng = np.random.default_rng(0)
    nnz_target = int(n_genes * n_cells * density)
    rows = rng.integers(0, n_genes, size=nnz_target, dtype=np.int32)
    cols = rng.integers(0, n_cells, size=nnz_target, dtype=np.int32)
    vals = rng.random(nnz_target, dtype=np.float32) * 5.0 + 0.1
    # Build genes×cells CSC directly.
    host_csc = sp.csc_matrix((vals, (rows, cols)), shape=(n_genes, n_cells)).astype(np.float32)
    host_csc.sort_indices()
    cu_csc = csp.csc_matrix(host_csc)
    print(f"[{label}] genes={n_genes} cells={n_cells} nnz={cu_csc.nnz}", flush=True)
    print(f"[{label}] uploading via _core.from_cupy_csr...", flush=True)
    device_mat = _core.from_cupy_csr(cu_csc)
    print(f"[{label}] device_mat OK, calling _core.pca(k={k})...", flush=True)
    try:
        result = _core.pca(device_mat, k, zero_center=True, scale=False, seed=0)
        print(f"[{label}] PASS — pca returned a result struct: {type(result).__name__}", flush=True)
        return True
    except Exception as e:
        print(f"[{label}] FAIL — {type(e).__name__}: {e}", flush=True)
        return False

# Progressive scale: small → medium → large
sizes = [
    ("tiny",   100,    50,  0.10),
    ("small",  1000,   500, 0.05),
    ("medium", 5000,  2000, 0.05),
    ("large", 20000, 10000, 0.05),
    ("xlarge",50000, 20000, 0.05),  # closest to test data 20866 × 310797
]

results = []
for label, m, n, dens in sizes:
    print(f"\n=== {label} {m} × {n} ===", flush=True)
    try:
        ok = try_pca(m, n, density=dens, k=10, label=label)
        results.append((label, m, n, "PASS" if ok else "FAIL"))
    except Exception as e:
        print(f"[{label}] HARNESS ERROR: {type(e).__name__}: {e}", flush=True)
        results.append((label, m, n, f"HARNESS-{type(e).__name__}"))

print("\n=== SUMMARY ===", flush=True)
for label, m, n, status in results:
    print(f"  {status:8s}  {label:8s}  ({m} × {n})", flush=True)
PYEOF

REPRO_EXIT=$?
echo ""
echo "REPRO_EXIT=${REPRO_EXIT}"

deactivate
rm -rf "$VENV"
date
exit 0
