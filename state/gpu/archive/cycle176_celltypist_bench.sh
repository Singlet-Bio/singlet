#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-176 Phase E — anno/celltypist medium-scale bench vs sklearn CPU.
#
# GPU kernel (anno::celltypist_predict): cuBLAS Sgemm (L = W^T . Z) +
#   bias-add kernel + strided softmax-argmax per cell.
# Citation: Dominguez Conde C et al. (2022) Science 376:eabl5197.
# Scales: 10k cells × 50 PCs, n_classes=20
#         30k cells × 50 PCs, n_classes=20
# Input:  synthetic fp32 Z/W/b (xorshift64 seed=42, untimed).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g008/g050/g051/g052).
#
# celltypist_predict() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: sklearn LogisticRegression.predict_proba (celltypist_ref.py):
#   Synthesized random weights (no training); only predict_proba(X) timed.
#   Same 2 warmup + 5 timed protocol; median wall time.
#   sklearn IS installed (CYCLE-172 confirmed, v1.6.1).
#
# §J.7 prediction: sklearn.predict_proba is BLAS-backed DGEMM + scipy softmax.
#   SOTA structure: BLAS-tight.  Expect class 3 (5-30×).
#   Could be 100-200× if sklearn/Python overhead dominates at small DGEMM scale.
#
# 30-min walltime. Output: state/cycle176_celltypist_bench_%j.log.
# Build: only bench_anno_celltypist_perf target.

#SBATCH --job-name=celltypist_bench_c176
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle176_celltypist_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle176_celltypist
GPU_LOG=/tmp/cycle176_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle176_sklearn_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-176: anno/celltypist Phase E bench ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# --- Python environment ------------------------------------------------------
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
echo "--- cmake build: bench_anno_celltypist_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_anno_celltypist_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: anno/celltypist (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_anno_celltypist_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- sklearn CPU baseline ----------------------------------------------------
echo ""
echo "=== SKLEARN CPU BASELINE: LogisticRegression.predict_proba (BLAS-backed) ==="
python3 "$SRC_DIR/bench/refs/celltypist_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-176 anno/celltypist Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  cublasSgemm (L = W^T . Z, n_classes x n_cells),"
echo "               ct_bias_add_kernel (L[k,c] += b[k]),"
echo "               ct_softmax_argmax_kernel (one block/cell, warp-shuffle)."
echo "  Algorithm:   CellTypist logistic-regression inference (Dominguez Conde 2022)."
echo "  Input:       synthetic fp32 Z/W/b, xorshift64 seed=42; 50 PCs, 20 classes."
echo ""
echo "  GPU results (singlet-gpu — anno::celltypist_predict):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  sklearn CPU baseline (LogisticRegression.predict_proba):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs sklearn wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | sklearn_wall_ms | ratio"
echo "  ------|-------------|-----------------|------"

# GPU CSV fields:    scale,n_cells,n_features,n_classes,wall_ms,mem_mb → field 5
# sklearn CSV fields: scale,n_cells,n_features,n_classes,wall_ms       → field 5
for SCALE in 10k 30k; do
    GPU_MS=$(grep "^${SCALE}," "$GPU_LOG" 2>/dev/null \
             | head -1 | cut -d',' -f5)
    PY_MS=$(grep "^${SCALE}," "$PY_LOG" 2>/dev/null \
            | head -1 | cut -d',' -f5)

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

    printf "  %-5s | %11s | %15s | %s\n" \
        "$SCALE" \
        "${GPU_MS:-N/A}" \
        "${PY_MS:-N/A}" \
        "$RATIO"
done

echo ""
echo "  Feature:    anno/celltypist (CYCLE-135)"
echo "  Citation:   Dominguez Conde C et al. (2022) Science 376:eabl5197"
echo "  Config:     n_features=50 PCs, n_classes=20, seed=42"
echo "  Note:       CPU baseline: sklearn LogisticRegression.predict_proba (BLAS-backed)"
echo "  Note:       §J.7 predicted class 3 / BLAS-tight (5-30×); possibly 100-200×"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
