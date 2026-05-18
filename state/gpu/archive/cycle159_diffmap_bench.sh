#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-159 Phase E — embed/diffmap medium-scale bench vs scanpy CPU.
# Scales: 10k + 30k cells x 50 PCs, dense, k=10, n_comps=5, synthetic.
# Algorithm: Coifman & Lafon 2005 diffusion map (cuSOLVER Ssyevd eigendecomp).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: g050 (H100 80 GB) or any other non-excluded GPU node.
# Dense transition matrix: n=30k → 30000^2*4B = 3.6 GB; requires >= 8 GB VRAM.

#SBATCH --job-name=diffmap_bench_c159
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle159_diffmap_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle159_diffmap
GPU_LOG=/tmp/cycle159_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle159_scanpy_${SLURM_JOB_ID}.csv

echo "=== CYCLE-159: embed/diffmap Phase E bench ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

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
echo "--- cmake build: bench_embed_diffmap_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_embed_diffmap_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: embed/diffmap (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_embed_diffmap_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scanpy CPU baseline -----------------------------------------------------
echo ""
echo "=== SCANPY CPU BASELINE: sc.tl.diffmap ==="
python3 "$SRC_DIR/bench/refs/diffmap_ref.py" 2>&1 | tee "$PY_LOG"
SCANPY_EXIT=$?
echo "SCANPY_EXIT=$SCANPY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-159 embed/diffmap bench"
echo "==================================================================="
echo ""
echo "  GPU results (singlet-gpu — diffmap only, kNN pre-built):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 || echo "  (no CSV rows found)"

echo ""
echo "  scanpy CPU baseline (sc.tl.diffmap, n_comps=5, n_neighbors=10):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio (GPU wall_ms vs scanpy wall_ms):"
# GPU CSV:    scale,n_cells,n_pcs,k,n_components,wall_ms,mem_mb  → field 6
# scanpy CSV: scale,n_cells,n_pcs,k,n_components,wall_ms         → field 6
GPU_10K=$(grep "^10k," "$GPU_LOG" 2>/dev/null | head -1 | cut -d',' -f6)
GPU_30K=$(grep "^30k," "$GPU_LOG" 2>/dev/null | head -1 | cut -d',' -f6)
SC_10K=$(grep  "^10k," "$PY_LOG"  2>/dev/null | head -1 | cut -d',' -f6)
SC_30K=$(grep  "^30k," "$PY_LOG"  2>/dev/null | head -1 | cut -d',' -f6)

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
echo "  Feature: embed/diffmap (CYCLE-150, Coifman & Lafon 2005)"
echo "  Config: n_comps=5, t=1, k=10, dense PCA input (50 PCs)"
echo "  Note: GPU timing covers diffmap only (kNN pre-built, untimed)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "SCANPY_EXIT=$SCANPY_EXIT"
echo "Date: $(date)"
