#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-174 Phase E — preprocess/magic medium-scale bench vs scipy SpMM CPU.
#
# GPU kernel (preprocess::magic_impute):
#   row_sum_graph_kernel (warp-shuffle per-row sums),
#   normalize_weights_kernel (D⁻¹W, binary-search row assignment),
#   csc_transpose_to_dense_kernel (scatter CSC X → dense Y₀),
#   cusparseSpMM × t=3 (ping-pong dense Y, CUSPARSE_SPMM_ALG_DEFAULT).
# Citation: van Dijk D et al. (2018) "Recovering Gene Interactions from
#           Single-Cell Data Using Data Diffusion." Cell 174:716-729.
#           https://doi.org/10.1016/j.cell.2018.05.061
# Scales: 10k cells × 5000 genes (nnz=2.5M, k=10, t=3)
#         30k cells × 5000 genes (nnz=7.5M, k=10, t=3)
# Input:  synthetic CSC (density=5%, counts 0–10), LCG seed=42 (untimed).
#         SNN graph built from synthetic 50-dim PCA embedding (untimed).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g050/g051/g052).
#
# magic_impute() timing: 2 warmup + 5 timed via cudaEvent (impute only).
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: scipy/numpy manual SpMM diffusion (magic_ref.py):
#   sklearn kNN (k=10) → symmetric adjacency → row-normalize Markov →
#   t=3 × (scipy.sparse CSR @ numpy dense float32).
#   Same 2 warmup + 5 timed protocol; median wall time.
#
# §J.7 prediction: cuSPARSE SpMM (3 iterations) vs scipy.sparse SpMM
#   (vectorized native C). scipy SpMM is tightly bound C (no Python per-call
#   overhead inside the SpMM kernel). Expect class 3 (10-30×) per CYCLE-163
#   wsum SpMM pattern. Magic multiplies by t=3 vs wsum's t=1; GPU scales
#   linearly with t; scipy does too — ratio should be similar.
#
# 30-min walltime. Output: state/cycle174_magic_bench_%j.log.
# Build: only bench_preprocess_magic_perf target.

#SBATCH --job-name=magic_bench_c174
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle174_magic_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle174_magic
GPU_LOG=/tmp/cycle174_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle174_scipy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-174: preprocess/magic Phase E bench ==="
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
echo "--- cmake build: bench_preprocess_magic_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_preprocess_magic_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: preprocess/magic (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_preprocess_magic_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scipy CPU baseline ------------------------------------------------------
echo ""
echo "=== SCIPY CPU BASELINE: sklearn kNN + row-normalize + t=3 scipy.sparse SpMM ==="
python3 "$SRC_DIR/bench/refs/magic_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-174 preprocess/magic Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  row_sum_graph_kernel (warp-shuffle),"
echo "               normalize_weights_kernel (D⁻¹W, binary-search row),"
echo "               csc_transpose_to_dense_kernel (scatter CSC → dense Y₀),"
echo "               cusparseSpMM × t=3 (CUSPARSE_SPMM_ALG_DEFAULT)."
echo "  Algorithm:   MAGIC graph diffusion imputation (van Dijk et al. 2018)."
echo "  Input:       synthetic CSC 5% density, seed=42; SNN k=10 (untimed)."
echo ""
echo "  GPU results (singlet-gpu — magic_impute):"
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

# GPU CSV fields:   scale,n_cells,n_genes,density,k,t,wall_ms,mem_mb → field 7
# scipy CSV fields: scale,n_cells,n_genes,density,k,t,wall_ms        → field 7
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
echo "  Feature:    preprocess/magic (CYCLE-124)"
echo "  Citation:   van Dijk et al. (2018) Cell 174:716-729"
echo "  Config:     n_genes=5000, density=5%, k=10, t=3, synthetic CSC seed=42"
echo "  Note:       CPU baseline: sklearn kNN → row-normalize → scipy.sparse SpMM ×3"
echo "  Note:       §J.7 predicted class 3 (10-30×): scipy SpMM is vectorized native C"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
