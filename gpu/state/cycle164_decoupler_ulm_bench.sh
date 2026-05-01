#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-164 Phase E — enrich/decoupler_ulm medium-scale bench vs numpy CPU.
#
# GPU kernel: 5 passes — per-cell mean (scatter+divide), per-pathway W stats
#   (fused warp-shuffle), var_W finalization, cuSPARSE SpMM (X^T · W), OLS
#   slope score kernel. Port of decoupleR ULM (Badia-i-Mompel et al. 2022).
# Citation: Badia-i-Mompel P et al. (2022) decoupleR, Bioinformatics Advances 2:vbac016.
# Scales: 10k cells × 5k genes (density 5%) and 30k cells × 5k genes.
# Method: ULM (Univariate Linear Model) — OLS regression slope β_1[c,p].
# Per §J.6 audit: NOT at risk of dense-n×n bug (SpMM O(nnz × p)).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g050/g051/g052).
#
# CPU baseline: manual numpy closed-form ULM implementation.
#   NOTE: `decoupler` Python package is NOT installed — baseline uses numpy/scipy.
#   score[c,p] = (XW[c,p]/m - mean_X[c]*mean_W[p]) / max(var_W[p], ε)
# Same input seed=42 as GPU bench.
#
# NOTE (CYCLE-158.1 pattern): if scipy not available in SLURM Python env,
# the orchestrator runs the ref locally and pastes CSV manually.
#
# 30-min walltime. Output: state/cycle164_decoupler_ulm_bench_%j.log.
# Build: only bench_enrich_decoupler_ulm_perf target.

#SBATCH --job-name=dulm_bench_c164
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle164_decoupler_ulm_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle164_dulm
GPU_LOG=/tmp/cycle164_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle164_numpy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-164: enrich/decoupler_ulm Phase E bench ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# --- Python environment ------------------------------------------------------
# Load Python 3.11 for numpy/scipy baseline (system Python 3.9 is past EOL).
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>/dev/null || true

# --- Build -------------------------------------------------------------------
echo ""
echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1 | tail -15
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED — stopping."; exit 1; }

echo ""
echo "--- cmake build: bench_enrich_decoupler_ulm_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_enrich_decoupler_ulm_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: enrich/decoupler_ulm (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_enrich_decoupler_ulm_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- numpy CPU baseline ------------------------------------------------------
echo ""
echo "=== NUMPY CPU BASELINE: ULM (manual closed-form, decoupler-equivalent) ==="
python3 "$SRC_DIR/bench/refs/decoupler_ulm_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-164 enrich/decoupler_ulm Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  5 passes (mean_X scatter, W stats fused, var_W, SpMM, score)"
echo "  Algorithm:   ULM/OLS slope β_1[c,p] = cov(X_c,W_p)/var(W_p)"
echo "               (decoupleR port, Badia-i-Mompel et al. 2022)"
echo "  §J.6 status: NOT at risk (SpMM O(nnz × p), no dense n×n)"
echo "  n_pathways:  50  |  density: 5%  |  seed: 42"
echo ""
echo "  GPU results (singlet-gpu — ULM):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  numpy CPU baseline:"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs numpy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | numpy_wall_ms | ratio"
echo "  ------|-------------|---------------|------"

# GPU CSV fields:   scale,n_cells,n_genes,density,n_pathways,wall_ms,mem_mb → field 6
# numpy CSV fields: scale,n_cells,n_genes,density,n_pathways,wall_ms        → field 6
for SCALE in 10k 30k; do
    GPU_MS=$(grep "^${SCALE}," "$GPU_LOG" 2>/dev/null \
             | head -1 | cut -d',' -f6)
    PY_MS=$(grep "^${SCALE}," "$PY_LOG" 2>/dev/null \
            | head -1 | cut -d',' -f6)

    RATIO="N/A"
    if [ -n "${GPU_MS:-}" ] && [ -n "${PY_MS:-}" ]; then
        RATIO=$(python3 -c "
g=float('${GPU_MS}'); s=float('${PY_MS}')
if g>0 and s>0:
    r=s/g
    print(f'{r:.2f}x (GPU faster)' if r>1 else f'{1/r:.2f}x (GPU slower)')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
    fi

    printf "  %-5s | %11s | %13s | %s\n" \
        "$SCALE" \
        "${GPU_MS:-N/A}" \
        "${PY_MS:-N/A}" \
        "$RATIO"
done

echo ""
echo "  Feature:    enrich/decoupler_ulm (CYCLE-130)"
echo "  Citation:   Badia-i-Mompel P et al. (2022) Bioinformatics Advances 2:vbac016"
echo "  Config:     n_pathways=50, density=5%, seed=42, no real .1pz needed"
echo "  Note:       CPU baseline is manual numpy (decoupler pkg not installed)"
echo "  Note:       Expected speedup ~10-30x (bimodal pattern per CYCLE-163 finding;"
echo "              numpy vectorized ULM is native BLAS, similar profile to scipy SpMM)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
