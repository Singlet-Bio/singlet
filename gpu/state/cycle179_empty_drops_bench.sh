#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-179 Phase E — qc/empty_drops medium-scale bench vs manual scipy/numpy CPU.
#
# GPU kernel (qc::empty_drops): 6 GPU passes —
#   Pass 1: UMI col-sum (one warp/droplet, cuRAND-free);
#   Pass 2a: mark empty (t_j <= lower);
#   Pass 2b: ambient scatter (atomic, one thread/nnz);
#   Pass 2c: normalize ambient → π + log_π;
#   Pass 3: observed LL (one warp/candidate);
#   Pass 4: MC p-values (one block/candidate, 256 threads, cuRAND Philox4x32,
#            shared-mem cumulative CDF + binary-search categorical sampling,
#            O(log m) per draw, t_j draws × niters iters distributed);
#   Pass 5: BH FDR (host-side sort+scan at D2H boundary);
#   Pass 6: scatter is_cell.
# Limits: m_genes <= 32768 (smem CDF); t_j > 100000 → p=1, skipped.
# Citation: Lun ATL et al. (2019) Genome Biology 20:63.
#
# Scales:
#   10k: n_droplets=10000, n_genes=3000, lower=100, niters=10000
#   30k: n_droplets=30000, n_genes=3000, lower=100, niters=10000
# Input: synthetic raw 10X-style sparse CSC (Zipf ambient pi, 80% empty
#         droplets with Poisson(30) UMI, 20% candidates with Poisson(200)+lower+1).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g008/g050/g051/g052).
#
# empty_drops() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: manual scipy/numpy (empty_drops_ref.py):
#   Passes 1-3: vectorized numpy CSC sum / ambient scatter / log_pi.
#   Pass 4: Python loop per candidate × niters multinomial draws
#            (rng.multinomial(t_j, pi) inside Python for-loop).
#   Pass 5: numpy argsort + BH scan (vectorized).
#   SOTA shape: per-candidate × per-iter Python loop with numpy multinomial.
#   python_overhead_multiplier = 10-100× (Python loop overhead per MC iter).
#   Same 2 warmup + 5 timed protocol; median wall time.
#
# §J.7 prediction (4-axis formula from §J.7 updated corpus):
#   python_overhead_multiplier = 10-100× (pure Python inner loop per MC iter per cand)
#   memory_bandwidth_advantage = 1× (no dense intermediates > 100 MB)
#   GPU compute class = medium-heavy (cuRAND init + smem CDF + binary search per draw)
#   Predicted speedup: 50-200× (class 2-3).
#
# 45-min walltime. Output: state/cycle179_empty_drops_bench_%j.log.
# Build: only bench_qc_empty_drops_perf target.

#SBATCH --job-name=empty_drops_bench_c179
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle179_empty_drops_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle179_empty_drops
GPU_LOG=/tmp/cycle179_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle179_numpy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-179: qc/empty_drops Phase E bench ==="
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
echo "--- cmake build: bench_qc_empty_drops_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_qc_empty_drops_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: qc/empty_drops (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_qc_empty_drops_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scipy/numpy CPU baseline ------------------------------------------------
echo ""
echo "=== SCIPY/NUMPY CPU BASELINE: manual emptyDrops (6-pass + MC loop) ==="
python3 "$SRC_DIR/bench/refs/empty_drops_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-179 qc/empty_drops Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  ed_col_sum_kernel (Pass 1),"
echo "               ed_mark_empty_kernel + ed_ambient_scatter_kernel (Pass 2ab),"
echo "               ed_normalize_ambient_kernel (Pass 2c),"
echo "               ed_obs_ll_kernel (Pass 3),"
echo "               ed_mc_pvalue_kernel (Pass 4: cuRAND Philox4x32 + smem CDF),"
echo "               BH FDR host-side sort+scan (Pass 5),"
echo "               ed_scatter_fdr_kernel + ed_pvalue_kernel (Pass 6)."
echo "  Algorithm:   DropletUtils::emptyDrops multinomial MC (Lun 2019)."
echo "  Input:       synthetic Zipf-ambient sparse CSC (80% empty, 20% candidates)."
echo "               n_genes=3000, lower=100, niters=10000, fdr_thresh=0.001."
echo ""
echo "  GPU results (singlet-gpu — qc::empty_drops):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  scipy/numpy CPU baseline (manual 6-pass with Python MC loop):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs numpy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | numpy_wall_ms | ratio"
echo "  ------|-------------|---------------|------"

# GPU CSV fields:   scale,n_droplets,n_genes,lower,niters,wall_ms,mem_mb → field 6
# numpy CSV fields: scale,n_droplets,n_genes,lower,niters,wall_ms       → field 6
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
echo "  Feature:    qc/empty_drops (CYCLE-134)"
echo "  Citation:   Lun ATL et al. (2019) Genome Biology 20:63"
echo "  Config:     n_genes=3000, lower=100, niters=10000, fdr_thresh=0.001"
echo "  Note:       CPU baseline: manual scipy/numpy with Python MC loop per candidate"
echo "  Note:       §J.7 predicted 50-200× (python_overhead_mult=10-100×,"
echo "              GPU compute=medium-heavy, memory_bw=1×)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
