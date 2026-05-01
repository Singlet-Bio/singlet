#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-172 Phase E — graph/kmeans medium-scale bench vs sklearn KMeans CPU.
#
# GPU kernel: Forgy init (host mt19937 random column copy); Lloyd loop:
#   km_col_norm_kernel (||C||² per centroid, warp-shuffle reduction),
#   cublasSgemm D[n×k] = X^T·C, km_assign_kernel (dist + argmin per cell),
#   km_count_changes_kernel (D2H scalar change count, convergence check),
#   km_scatter_kernel (atomic-scatter centroid update),
#   km_centroid_divide_kernel, km_inertia_kernel (final inertia).
# Citation: Lloyd (1982) "Least squares quantization in PCM." IEEE TIT 28:129-137.
# Scales: 10k cells × 50 PCs (k=10, max_iter=20)
#         30k cells × 50 PCs (k=10, max_iter=20)
# Method: Forgy init; Lloyd iterations with atomic-scatter centroid update.
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g050/g051/g052).
#
# Embedding: synthetic col-major fp32, LCG seed=42 (untimed).
# kmeans() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: sklearn KMeans(init='random', n_init=1, max_iter=20, algorithm='lloyd').
#   init='random' matches GPU Forgy; n_init=1 matches single-run bench.
#   Same 2 warmup + 5 timed protocol; median wall time.
#
# §J.7 prediction: kmeans is medium-compute (Sgemm distance dominates per iter;
#   max_iter * n * k * d FLOPs). sklearn KMeans is well-vectorized BLAS.
#   Expect class 3-mid (10-50×) per CYCLE-171 kbet finding for medium-compute kernels.
#
# NOTE (CYCLE-163/164/165/166/167/169/170/171 INFRA pattern): if sklearn not
# available in SLURM Python env, orchestrator runs the ref locally and
# pastes CSV manually.
#
# 30-min walltime. Output: state/cycle172_kmeans_bench_%j.log.
# Build: only bench_graph_kmeans_perf target.

#SBATCH --job-name=kmeans_bench_c172
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle172_kmeans_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle172_kmeans
GPU_LOG=/tmp/cycle172_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle172_sklearn_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-172: graph/kmeans Phase E bench ==="
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
echo "--- cmake build: bench_graph_kmeans_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_graph_kmeans_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: graph/kmeans (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_graph_kmeans_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- sklearn CPU baseline ----------------------------------------------------
echo ""
echo "=== SKLEARN CPU BASELINE: KMeans(init='random', n_init=1, max_iter=20) ==="
python3 "$SRC_DIR/bench/refs/kmeans_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-172 graph/kmeans Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  Forgy init (host mt19937); Lloyd loop:"
echo "               km_col_norm + cublasSgemm D=X^T·C + km_assign_kernel"
echo "               (dist+argmin, one block/cell) + km_scatter (atomic-scatter)"
echo "               + km_centroid_divide + km_inertia_kernel."
echo "  Algorithm:   Lloyd (1982) k-means; max_iter=20; tol_changes=0."
echo "  Embedding:   synthetic col-major fp32, LCG seed=42 (untimed)."
echo ""
echo "  GPU results (singlet-gpu — kmeans):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  sklearn CPU baseline:"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs sklearn wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | sklearn_wall_ms | ratio"
echo "  ------|-------------|-----------------|------"

# GPU CSV fields:     scale,n_cells,n_pcs,k,max_iter,wall_ms,mem_mb → field 6
# sklearn CSV fields: scale,n_cells,n_pcs,k,max_iter,wall_ms,...    → field 6
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

    printf "  %-5s | %11s | %15s | %s\n" \
        "$SCALE" \
        "${GPU_MS:-N/A}" \
        "${PY_MS:-N/A}" \
        "$RATIO"
done

echo ""
echo "  Feature:    graph/kmeans (CYCLE-149)"
echo "  Citation:   Lloyd (1982) IEEE TIT 28:129-137"
echo "  Config:     k=10, max_iter=20, Forgy init, seed=42, n_pcs=50"
echo "  Note:       CPU baseline is sklearn KMeans(init='random', n_init=1)"
echo "  Note:       §J.7 predicted class 3-mid (10-50×): Sgemm-dominated per-iter"
echo "              compute; sklearn KMeans uses well-vectorised BLAS internals."
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
