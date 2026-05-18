#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-187: build Python wheel + run 4 score_genes pytest tests from CYCLE-186.
# Validates singlet_gpu.enrich.run_score_genes wrapper (be0c343) end-to-end.
# ref: python/tests/test_enrichment.py::test_run_score_genes_*

#SBATCH --job-name=sg_c187_score_genes
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle187_score_genes_pytest_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc
export CMAKE_PREFIX_PATH="/opt/rh/gcc-toolset-13/root/usr/lib"

EIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3
export SKBUILD_CMAKE_ARGS="-DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR};-DEigen3_DIR=${EIGEN_INCLUDE_DIR}/cmake"
export CMAKE_ARGS="-DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR}"

# pyproject.toml requires Python >= 3.10; load 3.11 from Clipper module system
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>&1 || echo "WARN: module load failed; falling back to system python"

# Build artifacts + pip cache under /mnt/projects (home quota exhausted)
export PIP_CACHE_DIR=/mnt/projects/debruinz_project/singlet-gpu/.pip-cache
mkdir -p "$PIP_CACHE_DIR"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle187_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python
VENV=/tmp/singlet_gpu_venv_${SLURM_JOB_ID}

echo "=== CYCLE-187: score_genes wrapper pytest ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "Python: $(python --version)  (which: $(which python))"

# Bootstrap pip if the Spack module doesn't ship it
if ! python -m pip --version >/dev/null 2>&1; then
    echo "pip not available — bootstrapping via ensurepip..."
    python -m ensurepip --user --upgrade 2>&1 | tail -5
fi
echo "pip: $(python -m pip --version 2>&1)"

# ---------------------------------------------------------------------------
# Create venv
# ---------------------------------------------------------------------------
echo ""
echo "--- creating venv at ${VENV} ---"
python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# Install test dependencies (scanpy implies anndata + scipy)
# ---------------------------------------------------------------------------
echo ""
echo "--- installing test dependencies ---"
python -m pip install \
    "scikit-build-core>=0.9" \
    "pybind11>=2.11" \
    "pytest>=7.0" \
    "numpy>=1.24" \
    "scipy>=1.11" \
    "anndata>=0.10" \
    "scanpy" \
    2>&1 | tail -20

# Try cupy-cuda12x; if unavailable on this node, mark SKIP_CUPY and continue
echo ""
echo "--- installing cupy-cuda12x ---"
python -m pip install "cupy-cuda12x>=13.0" 2>&1 | tail -10
CUPY_EXIT=$?
if [ "$CUPY_EXIT" -ne 0 ]; then
    echo "WARN: cupy-cuda12x install failed (exit $CUPY_EXIT). GPU tests will SKIP via requires_gpu."
fi

# ---------------------------------------------------------------------------
# Build + install wheel (editable, no-isolation so venv pybind11/skbuild-core are used)
# ---------------------------------------------------------------------------
echo ""
echo "--- building + installing singlet_gpu wheel (editable, --no-build-isolation) ---"
rm -rf "$SKBUILD_BUILD_DIR"
cd "$PY_DIR"
python -m pip install -e . --no-build-isolation 2>&1 | tail -80
PIP_EXIT=${PIPESTATUS[0]}
echo "PIP_EXIT=${PIP_EXIT}"

if [ "${PIP_EXIT}" -ne 0 ]; then
    echo "=== pip install FAILED — aborting ==="
    deactivate
    rm -rf "$VENV"
    exit "${PIP_EXIT}"
fi

# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------
echo ""
echo "--- import smoke test ---"
python -c "
import singlet_gpu
print('singlet_gpu version:', singlet_gpu.__version__)
import singlet_gpu.enrich
print('singlet_gpu.enrich.run_score_genes:', callable(singlet_gpu.enrich.run_score_genes))
" 2>&1
IMPORT_EXIT=$?
echo "IMPORT_EXIT=${IMPORT_EXIT}"

# ---------------------------------------------------------------------------
# Run the 4 score_genes pytest tests
# ---------------------------------------------------------------------------
TEST_FILE="$PY_DIR/tests/test_enrichment.py"

declare -a TEST_NAMES=(
    "test_run_score_genes_basic"
    "test_run_score_genes_copy"
    "test_run_score_genes_missing_genes"
    "test_run_score_genes_vs_scanpy"
)
declare -a TEST_EXITS=()

cd "$PY_DIR/tests"
for T in "${TEST_NAMES[@]}"; do
    echo ""
    echo "--- pytest: ${T} ---"
    python -m pytest "$TEST_FILE::${T}" -v --tb=short 2>&1
    TEST_EXITS+=("${PIPESTATUS[0]}")
    echo "exit=${TEST_EXITS[-1]}"
done

# ---------------------------------------------------------------------------
# Cleanup venv
# ---------------------------------------------------------------------------
deactivate
rm -rf "$VENV"

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
echo ""
echo "=== SUMMARY ==="
echo "Build (pip install): $([ ${PIP_EXIT} -eq 0 ] && echo PASS || echo FAIL)"
echo "Import smoke:        $([ ${IMPORT_EXIT} -eq 0 ] && echo PASS || echo FAIL)"
for i in "${!TEST_NAMES[@]}"; do
    printf "%-45s %s\n" "${TEST_NAMES[i]}:" "$([ ${TEST_EXITS[i]} -eq 0 ] && echo PASS || echo FAIL)"
done

OVERALL=0
[ "${PIP_EXIT}" -ne 0 ] && OVERALL=1
for E in "${TEST_EXITS[@]}"; do [ "$E" -ne 0 ] && OVERALL=1; done
echo "Overall: $([ $OVERALL -eq 0 ] && echo PASS || echo FAIL)"
date
exit $OVERALL
