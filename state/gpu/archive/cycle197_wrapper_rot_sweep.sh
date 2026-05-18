#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-197: wrapper-rot SWEEP per §J.13.
#
# Runs ALL python/tests/test_*.py pytest tests in one SLURM job and produces a
# per-test pass/fail table.  Goal: identify which Python wrappers are healthy
# vs which are STUBS so we can triage rot at scale instead of one-at-a-time.
#
# Per §J.13: stub wrappers fail in N independent ways; running all tests in
# one job means we see ALL the failures at once and can group by root cause.

#SBATCH --job-name=sg_c197_wrapper_sweep
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle197_wrapper_sweep_%j.log

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
module load python/3.11.14 2>&1 || echo "WARN: module load failed; falling back"

export PIP_CACHE_DIR=/mnt/projects/debruinz_project/singlet-gpu/.pip-cache
mkdir -p "$PIP_CACHE_DIR"
export CMAKE_BUILD_PARALLEL_LEVEL=8
export SKBUILD_BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle197_skbuild

PY_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/python
VENV=/tmp/singlet_gpu_venv_${SLURM_JOB_ID}

echo "=== CYCLE-197: wrapper-rot SWEEP (all test_*.py at once) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

if ! python -m pip --version >/dev/null 2>&1; then
    python -m ensurepip --user --upgrade 2>&1 | tail -5
fi

echo ""
echo "--- creating venv at ${VENV} ---"
python -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet

echo ""
echo "--- installing test deps (pinned per CYCLE-189 §J.12) ---"
python -m pip install \
    "scikit-build-core>=0.9" \
    "pybind11>=2.11" \
    "pytest>=7.0" \
    "numpy>=1.24,<2.6" \
    "scipy>=1.11,<1.18" \
    "anndata>=0.10,<0.13" \
    "scanpy>=1.10,<1.12" \
    "cupy-cuda12x>=13.0,<15" \
    2>&1 | tail -10

echo ""
echo "--- pip install singlet_gpu (build wheel) ---"
python -m pip install -e "$PY_DIR" --no-build-isolation 2>&1 | tail -10
PIP_EXIT=$?
echo "PIP_EXIT=${PIP_EXIT}"

echo ""
echo "--- import smoke test ---"
python -c "
import singlet_gpu
print('singlet_gpu version:', singlet_gpu.__version__)
import singlet_gpu.preprocess, singlet_gpu.enrich, singlet_gpu.qc
import singlet_gpu.reduce, singlet_gpu.integrate, singlet_gpu.embed
print('all submodules imported OK')
" 2>&1
IMPORT_EXIT=$?
echo "IMPORT_EXIT=${IMPORT_EXIT}"

# ---------------------------------------------------------------------------
# Run ALL test files (excluding ones known to need extras like R or RAPIDS)
# ---------------------------------------------------------------------------
TEST_FILES=(
    test_bindings.py
    test_core.py
    test_io.py
    test_preprocess.py
    test_pp_neighbors.py
    test_reduce.py
    test_enrichment.py
    test_integrate.py
    test_de_pseudobulk.py
    test_lineage.py
    test_streaming.py
    test_tl_leiden.py
    test_tl_markers.py
    test_tl_rank_genes_groups.py
    test_tl_umap.py
    test_velocity.py
    test_new_features_smoke.py
)

cd "$PY_DIR/tests"

# Collect summary line per test_FILE — we run pytest --tb=line for compact output
SUMMARY_FILE=/tmp/cycle197_summary_${SLURM_JOB_ID}.txt
> "$SUMMARY_FILE"

for TF in "${TEST_FILES[@]}"; do
    if [ ! -f "$TF" ]; then
        echo "SKIP  $TF  (file not found)" >> "$SUMMARY_FILE"
        continue
    fi
    echo ""
    echo "============================================================"
    echo "--- pytest ${TF} ---"
    echo "============================================================"
    # --tb=line keeps output compact; -q hides per-test PASS verbosity.
    # We also use --no-header --no-summary except final.
    OUT=$(python -m pytest "$TF" --tb=line -q --no-header 2>&1)
    EXIT=$?
    echo "$OUT" | tail -40
    # Last non-empty line of pytest output is usually a summary like
    # "X passed, Y failed in Z.ZZs"
    LAST=$(echo "$OUT" | grep -E "^=" | tail -1)
    if [ -z "$LAST" ]; then LAST=$(echo "$OUT" | tail -1); fi
    printf "%-3s  %-32s  %s\n" "$([ $EXIT -eq 0 ] && echo PASS || echo FAIL)" "$TF" "$LAST" >> "$SUMMARY_FILE"
done

deactivate
rm -rf "$VENV"

echo ""
echo "============================================================"
echo "=== CYCLE-197 WRAPPER-ROT SWEEP SUMMARY ==="
echo "============================================================"
echo "Build (pip install): $([ ${PIP_EXIT} -eq 0 ] && echo PASS || echo FAIL)"
echo "Import smoke:        $([ ${IMPORT_EXIT} -eq 0 ] && echo PASS || echo FAIL)"
echo ""
echo "--- per-test_FILE results ---"
cat "$SUMMARY_FILE"
echo ""

# Overall: count PASS vs FAIL
N_PASS=$(grep -c "^PASS" "$SUMMARY_FILE")
N_FAIL=$(grep -c "^FAIL" "$SUMMARY_FILE")
N_SKIP=$(grep -c "^SKIP" "$SUMMARY_FILE")
echo "Test files: ${N_PASS} PASS, ${N_FAIL} FAIL, ${N_SKIP} SKIP"
date
# Always exit 0 so the sweep itself doesn't get marked failed; consumers grep
# the SUMMARY block for triage.
exit 0
