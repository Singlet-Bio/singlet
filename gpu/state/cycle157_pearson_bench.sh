#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-157 Phase E — preprocess/pearson_residuals medium-scale bench vs scanpy CPU.
# Scales: 10k + 30k cells × 5k genes, density 5%, synthetic CSC.
# Node: g003 (V100S sm_70-compatible).

#SBATCH --job-name=pearson_bench_c157
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g003
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle157_pearson_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle157_pearson
GPU_LOG=/tmp/cycle157_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle157_scanpy_${SLURM_JOB_ID}.csv

echo "=== CYCLE-157: preprocess/pearson_residuals Phase E bench ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# ─── Build ────────────────────────────────────────────────────────────────────
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
echo "--- cmake build: bench_preprocess_pearson_residuals_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_preprocess_pearson_residuals_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# ─── GPU bench ────────────────────────────────────────────────────────────────
echo ""
echo "=== GPU BENCH: pearson_residual_variance (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_preprocess_pearson_residuals_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# ─── scanpy CPU baseline ──────────────────────────────────────────────────────
echo ""
echo "=== SCANPY CPU BASELINE: pearson_residuals ==="
python3 "$SRC_DIR/bench/refs/pearson_residuals_ref.py" 2>&1 | tee "$PY_LOG"
SCANPY_EXIT=$?
echo "SCANPY_EXIT=$SCANPY_EXIT"

# ─── SUMMARY ─────────────────────────────────────────────────────────────────
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-157 pearson_residuals bench"
echo "==================================================================="
echo ""
echo "  GPU results (singlet-gpu, V100S g003):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 || echo "  (no CSV rows found)"

echo ""
echo "  scanpy CPU baseline (pearson_residuals, theta=100):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio (GPU wall_ms vs scanpy wall_ms):"
# Extract GPU 10k and 30k wall_ms from tee'd log (CSV format: scale,n_cells,n_genes,density,wall_ms,mem_mb)
GPU_10K=$(grep "^10k," "$GPU_LOG" 2>/dev/null | head -1 | cut -d',' -f5)
GPU_30K=$(grep "^30k," "$GPU_LOG" 2>/dev/null | head -1 | cut -d',' -f5)
# Extract scanpy 10k and 30k wall_ms (CSV format: scale,n_cells,n_genes,density,wall_ms)
SC_10K=$(grep "^10k," "$PY_LOG" 2>/dev/null | head -1 | cut -d',' -f5)
SC_30K=$(grep "^30k," "$PY_LOG" 2>/dev/null | head -1 | cut -d',' -f5)

echo "  scale | GPU_wall_ms | scanpy_wall_ms | speedup"
echo "  ------|-------------|----------------|--------"
if [ -n "${GPU_10K:-}" ] && [ -n "${SC_10K:-}" ]; then
    RATIO_10K=$(python3 -c "g=${GPU_10K};s=${SC_10K}; print(f'{s/g:.1f}x' if g>0 else 'N/A')" 2>/dev/null || echo "N/A")
    printf "  %-5s | %11s | %14s | %s\n" "10k" "$GPU_10K" "$SC_10K" "$RATIO_10K"
else
    printf "  %-5s | %11s | %14s | %s\n" "10k" "${GPU_10K:-N/A}" "${SC_10K:-N/A}" "N/A"
fi
if [ -n "${GPU_30K:-}" ] && [ -n "${SC_30K:-}" ]; then
    RATIO_30K=$(python3 -c "g=${GPU_30K};s=${SC_30K}; print(f'{s/g:.1f}x' if g>0 else 'N/A')" 2>/dev/null || echo "N/A")
    printf "  %-5s | %11s | %14s | %s\n" "30k" "$GPU_30K" "$SC_30K" "$RATIO_30K"
else
    printf "  %-5s | %11s | %14s | %s\n" "30k" "${GPU_30K:-N/A}" "${SC_30K:-N/A}" "N/A"
fi

echo ""
echo "  Prior small-scale result (pareto-frontier.md CYCLE-118): 0.269 ms GPU vs 3388.6 ms scanpy = 12609x"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "SCANPY_EXIT=$SCANPY_EXIT"
echo "Date: $(date)"
