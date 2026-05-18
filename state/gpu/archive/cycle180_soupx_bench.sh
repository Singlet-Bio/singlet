#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-180 Phase E — qc/soupx medium-scale bench vs manual scipy/numpy CPU.
#
# GPU kernel (qc::soupx): 5 GPU passes —
#   Pass 1: per-droplet UMI sum t[j] (warp/col CSC);
#   Pass 2: ambient scatter (atomic/nnz for empty cols, t[j] <= lower) +
#           normalize → pi[g] (CUB DeviceReduce scalar + kernel);
#   Pass 3: top-ambient-gene mask (one D2H of pi, host partial_sort + threshold,
#           H2D mask uint8);
#   Pass 4: per-cell rho_c (warp/col, top_amb_mask numerator / denom_pi);
#   Pass 5: dense corrected output (cudaMemset + nnz-overwrite;
#           max(0, x - rho_c * t_c * pi[g]) for stored nnz;
#           algebraic identity max(0, 0 - ...) = 0 from cudaMemset).
# Memory guard: throws if m*n*4 > 0.5 * free GPU memory.
# Citation: Young MD, Behjati S (2020) GigaScience 9:giaa151.
#
# Scales:
#   10k: n_droplets=10000, n_genes=3000, lower=100
#   30k: n_droplets=30000, n_genes=3000, lower=100
# Input: synthetic raw 10X-style sparse CSC (Zipf ambient pi, 80% empty
#         droplets with Poisson(30) UMI, 20% cells with Poisson(200)+lower+1).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g008/g050/g051/g052).
#
# soupx() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: manual scipy/numpy (soupx_ref.py):
#   Pass 1: X.sum(axis=0) (vectorized CSC)
#   Pass 2: X[:, is_empty].sum(axis=1) / total (vectorized CSC slice)
#   Pass 3: np.argpartition(pi, -n_top) (vectorized)
#   Pass 4: X[top_mask, :].sum(axis=0) / (t * denom_pi) (vectorized CSC slice)
#   Pass 5: COO nnz-iterate: corrected = max(0, x - rho*t*pi) (vectorized arrays)
#   SOTA shape: fully vectorized numpy/scipy — no Python inner loops.
#   python_overhead_multiplier = 1× (tight vectorized ops, no per-element Python).
#   Same 2 warmup + 5 timed protocol; median wall time.
#
# §J.7 prediction (4-axis formula from §J.7 updated corpus):
#   python_overhead_multiplier = 1× (fully vectorized scipy/numpy, no loops)
#   memory_bandwidth_advantage = 1× (CPU stays sparse; GPU dense write is a wash)
#   GPU compute class = light-medium (5 passes: warp colsum, atomic scatter,
#     topK mask, warp rho, dense memset + nnz-overwrite — no BLAS/cuRAND)
#   Predicted speedup: 10-30× (class 3, same as wsum/decoupler family).
#
# 30-min walltime. Output: state/cycle180_soupx_bench_%j.log.
# Build: only bench_qc_soupx_perf target.

#SBATCH --job-name=soupx_bench_c180
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle180_soupx_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle180_soupx
GPU_LOG=/tmp/cycle180_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle180_numpy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-180: qc/soupx Phase E bench ==="
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
echo "--- cmake build: bench_qc_soupx_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_qc_soupx_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: qc/soupx (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_qc_soupx_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scipy/numpy CPU baseline ------------------------------------------------
echo ""
echo "=== SCIPY/NUMPY CPU BASELINE: manual SoupX (5-pass vectorized numpy) ==="
python3 "$SRC_DIR/bench/refs/soupx_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-180 qc/soupx Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  sx_col_sum_kernel (Pass 1),"
echo "               sx_ambient_scatter_kernel + sx_normalize_ambient_kernel (Pass 2),"
echo "               host topK mask + H2D (Pass 3),"
echo "               sx_rho_kernel (Pass 4),"
echo "               cudaMemset + sx_correct_kernel (Pass 5)."
echo "  Algorithm:   SoupX ambient RNA correction (Young & Behjati 2020)."
echo "  Input:       synthetic Zipf-ambient sparse CSC (80% empty, 20% cells)."
echo "               n_genes=3000, lower=100, top_ambient_frac=0.10."
echo ""
echo "  GPU results (singlet-gpu — qc::soupx):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  scipy/numpy CPU baseline (manual 5-pass vectorized):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs numpy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | numpy_wall_ms | ratio"
echo "  ------|-------------|---------------|------"

# GPU CSV fields:   scale,n_droplets,n_genes,lower,wall_ms,mem_mb → field 5
# numpy CSV fields: scale,n_droplets,n_genes,lower,wall_ms        → field 5
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
echo "  Feature:    qc/soupx (CYCLE-141)"
echo "  Citation:   Young MD, Behjati S (2020) GigaScience 9:giaa151"
echo "  Config:     n_genes=3000, lower=100, top_ambient_frac=0.10"
echo "  Note:       CPU baseline: fully vectorized scipy/numpy (no Python inner loops)"
echo "  Note:       §J.7 predicted 10-30× (python_overhead_mult=1×,"
echo "              GPU compute=light-medium, memory_bw=1× [GPU dense vs CPU sparse])"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
