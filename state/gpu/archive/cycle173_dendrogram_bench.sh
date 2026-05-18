#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-173 Phase E — embed/dendrogram medium-scale bench vs scipy/numpy CPU.
#
# GPU kernel (embed::dendrogram):
#   dgram_centroid_scatter_kernel (atomic-scatter, one block/cell),
#   dgram_centroid_divide_kernel, dgram_col_mean_kernel, dgram_center_kernel,
#   dgram_col_norm_kernel, dgram_normalize_kernel,
#   cublasSgemm corr = μ_norm^T · μ_norm (k × k), dgram_dist_kernel d = 1 - corr.
#   Host UPGMA O(k³) pure C++ (k ≤ 50 → trivial).
# Citation: Wolf et al. (2018) "SCANPY: large-scale single-cell gene expression
#           data analysis." Genome Biol 19:15. doi:10.1186/s13059-017-1382-0.
# Scales: 10k cells × 50 PCs (k=20, round-robin labels)
#         30k cells × 50 PCs (k=20, round-robin labels)
# Input:  dense CSC (density=1.0, nnz = n_genes × n_cells, synthetic).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g050/g051/g052).
#
# Embedding: dense synthetic CSC (counts 0–10), LCG seed=42 (untimed).
# dendrogram() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: scipy/numpy full pipeline:
#   np.add.at centroid scatter + center + L2-norm + scipy.spatial.distance.pdist
#   (cosine) + scipy.cluster.hierarchy.linkage(method='average').
#   Same 2 warmup + 5 timed protocol; median wall time.
#
# §J.7 prediction: dendrogram is dominated by per-cell centroid scatter (light
#   per-cell, bounded by n_genes=50) + trivial k×k Sgemm (k=20) + host UPGMA
#   (~8000 ops). Total GPU should be sub-ms. scipy has Python orchestration
#   overhead on top of C routines. Expect class 2 (50-200×) speedup.
#
# NOTE (CYCLE-163/164/165/166/167/169/170/171/172 INFRA pattern): if scipy not
# available in SLURM Python env, orchestrator runs the ref locally and
# pastes CSV manually.
#
# 30-min walltime. Output: state/cycle173_dendrogram_bench_%j.log.
# Build: only bench_embed_dendrogram_perf target.

#SBATCH --job-name=dendrogram_bench_c173
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle173_dendrogram_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle173_dendrogram
GPU_LOG=/tmp/cycle173_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle173_scipy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-173: embed/dendrogram Phase E bench ==="
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
echo "--- cmake build: bench_embed_dendrogram_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_embed_dendrogram_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: embed/dendrogram (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_embed_dendrogram_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scipy CPU baseline ------------------------------------------------------
echo ""
echo "=== SCIPY CPU BASELINE: centroid scatter + pdist(cosine) + linkage(average) ==="
python3 "$SRC_DIR/bench/refs/dendrogram_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-173 embed/dendrogram Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  dgram_centroid_scatter (atomic-scatter, one block/cell),"
echo "               dgram_centroid_divide, dgram_col_mean, dgram_center,"
echo "               dgram_col_norm, dgram_normalize,"
echo "               cublasSgemm corr = μ_norm^T · μ_norm (k × k),"
echo "               dgram_dist d = 1 - corr; host UPGMA O(k³)."
echo "  Algorithm:   scanpy.tl.dendrogram UPGMA hierarchical clustering."
echo "  Embedding:   dense CSC fp32 (50 PCs, counts 0–10), seed=42 (untimed)."
echo ""
echo "  GPU results (singlet-gpu — dendrogram):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  scipy CPU baseline:"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs scipy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | scipy_wall_ms | ratio"
echo "  ------|-------------|---------------|------"

# GPU CSV fields:   scale,n_cells,n_pcs,n_clusters,wall_ms,mem_mb → field 5
# scipy CSV fields: scale,n_cells,n_pcs,n_clusters,wall_ms        → field 5
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

    printf "  %-5s | %11s | %13s | %s\n" \
        "$SCALE" \
        "${GPU_MS:-N/A}" \
        "${PY_MS:-N/A}" \
        "$RATIO"
done

echo ""
echo "  Feature:    embed/dendrogram (CYCLE-146)"
echo "  Citation:   Wolf et al. (2018) Genome Biol 19:15"
echo "  Config:     k=20, n_pcs=50, dense CSC (density=1.0), round-robin labels"
echo "  Note:       CPU baseline: scipy centroid scatter + pdist(cosine) + linkage(average)"
echo "  Note:       §J.7 predicted class 2 (50-200×): sub-ms GPU (scatter+tiny Sgemm+host UPGMA)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
