#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-170 Phase E — integrate/asw medium-scale bench vs numpy/sklearn CPU.
#
# GPU kernel: one block per cell; shared-mem float smem_sum[n_labels] and
#   int smem_cnt[n_labels]; thread 0 sweeps kNN row accumulating distances
#   by label; computes a (intra-cluster mean dist), b (min other-cluster
#   mean dist), silhouette = (b - a) / max(a, b).
#   cub::DeviceReduce::Sum collapses per-cell scores to scalar ASW mean.
# Citation: Rousseeuw PJ (1987) J Comput Appl Math 20:53-65.
#           Luecken et al. (2022) Nature Methods 19:41-50 (scIB benchmarking).
# Scales: 10k cells × 50 PCs (k=10, n_clusters=4)
#         30k cells × 50 PCs (k=15, n_clusters=4)
# Method: per-cell a/b from kNN distances; round-robin cluster labels.
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g050/g051/g052).
#
# kNN build: untimed warmup (Exact backend, L2, k=10/15).
# ASW timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
# Cluster labels: round-robin label[c] = c % 4.
#
# CPU baseline: manual numpy vectorised ASW.
#   sklearn.neighbors.NearestNeighbors (brute, Euclidean, k=10/15).
#   one_hot distances per label, mean a/b, silhouette = (b-a)/max(a,b).
#   Vectorised via numpy broadcasting (no Python loop over cells).
#
# NOTE (CYCLE-163/164/165/166/167 INFRA pattern): if sklearn not available in
# SLURM Python env, orchestrator runs the ref locally and pastes CSV manually.
#
# 30-min walltime. Output: state/cycle170_asw_bench_%j.log.
# Build: only bench_integrate_asw_perf target.

#SBATCH --job-name=asw_bench_c170
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle170_asw_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle170_asw
GPU_LOG=/tmp/cycle170_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle170_numpy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-170: integrate/asw Phase E bench ==="
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
echo "--- cmake build: bench_integrate_asw_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_integrate_asw_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: integrate/asw (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_integrate_asw_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- numpy/sklearn CPU baseline ----------------------------------------------
echo ""
echo "=== NUMPY/SKLEARN CPU BASELINE: ASW (vectorised per-cell a/b) ==="
python3 "$SRC_DIR/bench/refs/asw_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-170 integrate/asw Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  one block/cell; shared-mem float smem_sum[n_labels] +"
echo "               int smem_cnt[n_labels]; thread 0 serial scan (deterministic)."
echo "  Algorithm:   ASW — per-cell a (intra-cluster mean dist), b (min other-"
echo "               cluster mean dist); silhouette = (b-a)/max(a,b) ∈ [-1,1]."
echo "               Scalar ASW = mean via cub::DeviceReduce::Sum."
echo "  kNN:         Exact backend (L2, k=10/15) — untimed warmup."
echo "  Cluster labels: round-robin label[c] = c % 4."
echo ""
echo "  GPU results (singlet-gpu — ASW):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  numpy/sklearn CPU baseline:"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs numpy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | numpy_wall_ms | ratio"
echo "  ------|-------------|---------------|------"

# GPU CSV fields:   scale,n_cells,n_pcs,k,n_clusters,wall_ms,mem_mb → field 6
# numpy CSV fields: scale,n_cells,n_pcs,k,n_clusters,wall_ms        → field 6
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
echo "  Feature:    integrate/asw (CYCLE-139)"
echo "  Citation:   Rousseeuw PJ (1987) J Comput Appl Math 20:53-65"
echo "              Luecken et al. (2022) Nature Methods 19:41-50 (scIB)"
echo "  Config:     k=10/15, n_clusters=4, round-robin labels, n_pcs=50"
echo "  Note:       CPU baseline is numpy vectorised ASW (sklearn kNN, not timed)"
echo "  Note:       Expected speedup class 1-2 (~100-300x): ASW is per-cell"
echo "              intra/inter-cluster hist on GPU vs numpy vectorised broadcasting."
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
