#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-161 Phase E — embed/dpt medium-scale bench vs scanpy CPU.
# Scale: 10k cells x 50 PCs, dense, k=10, n_eigenvecs=15, synthetic.
# Algorithm: Haghverdi et al. 2016 DPT (cuSOLVER Ssyevd eigendecomp).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: g050 (H100 80 GB) or any other non-excluded GPU node.
#
# §J.6 AT-RISK hypothesis test: dpt uses the same dense n×n + Ssyevd pattern
# that caused 14× slowdown + crash in diffmap (CYCLE-159). Testing whether
# dpt exhibits the same scaling failure at n=10k.
# Prediction: GPU dpt will be 5-20× slower than sc.tl.dpt (sparse ARPACK).
# If confirmed: file combined diffmap+dpt sparse-eigensolver rewrite.
# If refuted: update §J.6 audit — dpt algorithm may differ subtly.
#
# 30k is SKIPPED for GPU (same cuSOLVER pattern that crashed diffmap at 30k).
# Scanpy 30k DPT is included in the Python ref (ARPACK, tractable ~15-30s).

#SBATCH --job-name=dpt_bench_c161
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle161_dpt_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle161_dpt
GPU_LOG=/tmp/cycle161_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle161_scanpy_${SLURM_JOB_ID}.csv

echo "=== CYCLE-161: embed/dpt Phase E bench (§J.6 AT-RISK hypothesis test) ==="
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
echo "--- cmake build: bench_embed_dpt_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_embed_dpt_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: embed/dpt (singlet-gpu, 10k cells, n_eigenvecs=15) ==="
"$BUILD_DIR/bench/bench_embed_dpt_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scanpy CPU baseline -----------------------------------------------------
echo ""
echo "=== SCANPY CPU BASELINE: sc.tl.dpt (10k + 30k cells, iroot=0) ==="
python3 "$SRC_DIR/bench/refs/dpt_ref.py" 2>&1 | tee "$PY_LOG"
SCANPY_EXIT=$?
echo "SCANPY_EXIT=$SCANPY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-161 embed/dpt bench (§J.6 AT-RISK test)"
echo "==================================================================="
echo ""
echo "  Hypothesis: GPU dpt at 10k will be 5-20× SLOWER than scanpy ARPACK."
echo "  (Same dense-n×n + cusolverDnSsyevd pattern as diffmap, CYCLE-159.)"
echo ""
echo "  GPU results (singlet-gpu — dpt only, kNN pre-built):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 || echo "  (no CSV rows found)"

echo ""
echo "  scanpy CPU baseline (sc.tl.dpt, n_comps=15, n_neighbors=10, iroot=0):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio (GPU wall_ms vs scanpy wall_ms, 10k):"
# GPU CSV:    scale,n_cells,n_pcs,k,n_eigenvecs,wall_ms,mem_mb → field 6
# scanpy CSV: scale,n_cells,n_pcs,k,n_eigenvecs,wall_ms        → field 6
GPU_10K=$(grep "^10k," "$GPU_LOG" 2>/dev/null | head -1 | cut -d',' -f6)
SC_10K=$(grep  "^10k," "$PY_LOG"  2>/dev/null | head -1 | cut -d',' -f6)
SC_30K=$(grep  "^30k," "$PY_LOG"  2>/dev/null | head -1 | cut -d',' -f6)

echo "  scale | GPU_wall_ms | scanpy_wall_ms | speedup(scanpy/GPU)"
echo "  ------|-------------|----------------|--------------------"
if [ -n "${GPU_10K:-}" ] && [ -n "${SC_10K:-}" ]; then
    RATIO_10K=$(python3 -c "g=${GPU_10K};s=${SC_10K}; print(f'{s/g:.2f}x (GPU faster)' if g>0 and s/g>1 else f'{g/s:.2f}x (GPU SLOWER)' if g>0 and s>0 else 'N/A')" 2>/dev/null || echo "N/A")
    printf "  %-5s | %11s | %14s | %s\n" "10k" "$GPU_10K" "$SC_10K" "$RATIO_10K"
else
    printf "  %-5s | %11s | %14s | %s\n" "10k" "${GPU_10K:-N/A}" "${SC_10K:-N/A}" "N/A"
fi
if [ -n "${SC_30K:-}" ]; then
    printf "  %-5s | %11s | %14s | %s\n" "30k" "SKIPPED" "$SC_30K" "N/A (GPU skipped)"
fi

echo ""
echo "  §J.6 verdict:"
if [ -n "${GPU_10K:-}" ] && [ -n "${SC_10K:-}" ]; then
    python3 - <<'PYEOF'
import sys, os
try:
    gpu = float(os.environ.get('GPU_10K_VAL', '0'))
    sc  = float(os.environ.get('SC_10K_VAL', '0'))
except:
    sys.exit(0)
if gpu <= 0 or sc <= 0:
    print("  Cannot compute ratio (one side errored).")
    sys.exit(0)
ratio = gpu / sc  # >1 means GPU is slower
if ratio >= 5:
    print(f"  CONFIRMED: GPU is {ratio:.1f}x SLOWER than scanpy ARPACK at 10k.")
    print("  Action: file combined diffmap+dpt sparse-eigensolver rewrite (CYCLE-159.1 + DPT).")
elif ratio <= 0.5:
    print(f"  REFUTED: GPU is {1/ratio:.1f}x FASTER than scanpy at 10k.")
    print("  Action: update §J.6 audit — dpt does NOT hit the Ssyevd bottleneck at 10k.")
else:
    print(f"  BORDERLINE: GPU is {ratio:.1f}x slower than scanpy at 10k (hypothesis: 5-20x).")
    print("  Action: investigate — may need larger scale or profiling to confirm/refute.")
PYEOF
    GPU_10K_VAL="$GPU_10K" SC_10K_VAL="$SC_10K" python3 - <<'PYEOF2'
import sys, os
try:
    gpu = float(os.environ['GPU_10K_VAL'])
    sc  = float(os.environ['SC_10K_VAL'])
except:
    sys.exit(0)
if gpu <= 0 or sc <= 0:
    print("  Cannot compute ratio (one side errored).")
    sys.exit(0)
ratio = gpu / sc
if ratio >= 5:
    print(f"  CONFIRMED: GPU is {ratio:.1f}x SLOWER than scanpy ARPACK at 10k.")
    print("  Action: file combined diffmap+dpt sparse-eigensolver rewrite (CYCLE-159.1 + DPT).")
elif ratio <= 0.5:
    print(f"  REFUTED: GPU is {1/ratio:.1f}x FASTER than scanpy at 10k.")
    print("  Action: update §J.6 audit — dpt does NOT hit the Ssyevd bottleneck at 10k.")
else:
    print(f"  BORDERLINE: GPU is {ratio:.1f}x slower than scanpy at 10k (hypothesis: 5-20x).")
    print("  Action: investigate further (profile Pass 5 cuSOLVER Ssyevd vs ARPACK).")
PYEOF2
fi

echo ""
echo "  Feature: embed/dpt (CYCLE-142, Haghverdi et al. 2016)"
echo "  Config: n_eigenvecs=15, root_cell=0, k=10, dense PCA input (50 PCs)"
echo "  Note: GPU timing covers dpt only (kNN pre-built, untimed)"
echo "  Note: scanpy timing covers sc.tl.dpt only (neighbors+diffmap untimed)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "SCANPY_EXIT=$SCANPY_EXIT"
echo "Date: $(date)"
