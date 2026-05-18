#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-178 Phase E — anno/symphony medium-scale bench vs manual NumPy CPU.
#
# GPU kernel (anno::symphony_predict): 5 sequential device passes —
#   standardize (element-wise kernel), PCA project (cuBLAS Sgemm),
#   distance to centroids (sumsq kernels + Sgemm cross + combine kernel),
#   soft assignment (1/d + warp-shuffle L1-norm per cell),
#   label transfer (cuBLAS Sgemm) + argmax kernel.
# Citation: Kang JB, Nathan A, Weinand K et al. (2021) Nat Commun 12:5890.
# Pairs with: anno/celltypist (CYCLE-176) for cross-comparability.
#
# Scales:
#   10k cells × 50 PCs, n_centroids=20, n_classes=20
#   30k cells × 50 PCs, n_centroids=20, n_classes=20
# Input: synthetic fp32 Z/W_pca/C_ref/P_label/mu/sigma (xorshift64 seed=42, untimed).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g008/g050/g051/g052).
#
# symphony_predict() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: manual NumPy symphony (symphony_ref.py):
#   5-pass vectorized numpy — standardize + PCA project (np.dot = Sgemm) +
#   distance (broadcasting) + soft-assign + label transfer (np.dot).
#   No Python per-cell loop; no Python Symphony package needed.
#   Same 2 warmup + 5 timed protocol; median wall time.
#
# §J.7 prediction (4-axis formula from §J.7 §J.7 CYCLE-177):
#   SOTA shape: vectorized numpy (tight BLAS + broadcasting), no per-batch loops.
#     python_overhead_multiplier = 1× (single tight call, no per-cell Python).
#   Dense intermediates at 30k × 50: ~6 MB max → memory_bandwidth_advantage = 1×.
#   GPU compute: medium/heavy (5 passes vs celltypist 3).
#   Prediction: 30-100× class (similar to celltypist 50× but slightly lower
#     because GPU has 2 extra passes; numpy distance+softmax also vectorized).
#
# 30-min walltime. Output: state/cycle178_symphony_bench_%j.log.
# Build: only bench_anno_symphony_perf target.

#SBATCH --job-name=symphony_bench_c178
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle178_symphony_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle178_symphony
GPU_LOG=/tmp/cycle178_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle178_numpy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-178: anno/symphony Phase E bench ==="
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
echo "--- cmake build: bench_anno_symphony_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_anno_symphony_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: anno/symphony (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_anno_symphony_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- NumPy CPU baseline -------------------------------------------------------
echo ""
echo "=== NUMPY CPU BASELINE: manual Symphony (5-pass vectorized numpy) ==="
python3 "$SRC_DIR/bench/refs/symphony_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-178 anno/symphony Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  sy_standardize_kernel (Pass 1),"
echo "               cublasSgemm PCA project (Pass 2),"
echo "               sy_col_sumsq + sy_row_sumsq + cublasSgemm cross + sy_dist_combine (Pass 3),"
echo "               sy_soft_assign_kernel (Pass 4),"
echo "               cublasSgemm label transfer + sy_argmax_conf_kernel (Pass 5)."
echo "  Algorithm:   Symphony centroid-projection reference mapping (Kang 2021)."
echo "  Input:       synthetic fp32 Z/W_pca/C_ref/P_label/mu/sigma, xorshift64 seed=42."
echo "               50 PCs, 20 centroids, 20 classes."
echo ""
echo "  GPU results (singlet-gpu — anno::symphony_predict):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  NumPy CPU baseline (manual 5-pass vectorized numpy):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs NumPy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | numpy_wall_ms | ratio"
echo "  ------|-------------|---------------|------"

# GPU CSV fields:   scale,n_cells,n_features,k_pcs,n_centroids,n_classes,wall_ms,mem_mb → field 7
# NumPy CSV fields: scale,n_cells,n_features,k_pcs,n_centroids,n_classes,wall_ms       → field 7
for SCALE in 10k 30k; do
    GPU_MS=$(grep "^${SCALE}," "$GPU_LOG" 2>/dev/null \
             | head -1 | cut -d',' -f7)
    PY_MS=$(grep "^${SCALE}," "$PY_LOG" 2>/dev/null \
            | head -1 | cut -d',' -f7)

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
echo "  Feature:    anno/symphony (CYCLE-138)"
echo "  Citation:   Kang JB, Nathan A, Weinand K et al. (2021) Nat Commun 12:5890"
echo "  Config:     n_features=50 PCs, n_centroids=20, n_classes=20, seed=42"
echo "  Note:       CPU baseline: manual NumPy 5-pass vectorized (no Python Symphony pkg)"
echo "  Note:       §J.7 predicted 30-100× class (BLAS-tight numpy, 5 GPU passes)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
