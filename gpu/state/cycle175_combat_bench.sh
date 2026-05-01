#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-175 Phase E — integrate/combat medium-scale bench vs scanpy.pp.combat CPU.
#
# GPU kernel (integrate::combat): 7 device passes, in-place Z-buffer reuse,
#   EB shrinkage at max_iter=2.
# Citation: Johnson WE, Li C, Rabinovic A (2007) Biostatistics 8:118-127.
# Scales: 10k cells × 5000 genes (dense, 4 batches, max_iter=2)
#         30k cells × 5000 genes (dense, 4 batches, max_iter=2)
# Input:  synthetic dense CSC (counts 0–20, LCG seed=42, untimed).
#         batch labels: round-robin 0..3 across n_cells (untimed).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g008/g050/g051/g052).
#
# combat() timing: 2 warmup + 5 timed via cudaEvent.
# Memory: bench::PeakMemTracker (cudaMemGetInfo delta).
#
# CPU baseline: scanpy.pp.combat (combat_ref.py):
#   scanpy.pp.combat(adata, key='batch') — vectorised NumPy/SciPy EB.
#   Same 2 warmup + 5 timed protocol; median wall time.
#   Note: scanpy.pp.combat works on dense adata.X — fair comparison.
#
# §J.7 prediction: scanpy.pp.combat is vectorised numpy (closed-form EB,
#   no per-gene Python loop).  SOTA structure: "vectorized numpy".
#   Expect class 2-3 (50-300×).  Could be class 1 (10-50×) if dense m×n
#   ops dominate equally on CPU and GPU.
#
# 45-min walltime. Output: state/cycle175_combat_bench_%j.log.
# Build: only bench_integrate_combat_perf target.

#SBATCH --job-name=combat_bench_c175
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle175_combat_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle175_combat
GPU_LOG=/tmp/cycle175_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle175_scanpy_${SLURM_JOB_ID}.csv

mkdir -p "$BUILD_DIR"

echo "=== CYCLE-175: integrate/combat Phase E bench ==="
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
echo "--- cmake build: bench_integrate_combat_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_integrate_combat_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: integrate/combat (singlet-gpu) ==="
"$BUILD_DIR/bench/bench_integrate_combat_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scanpy CPU baseline -----------------------------------------------------
echo ""
echo "=== SCANPY CPU BASELINE: scanpy.pp.combat (parametric EB, vectorised numpy) ==="
python3 "$SRC_DIR/bench/refs/combat_ref.py" 2>&1 | tee "$PY_LOG"
PY_EXIT=$?
echo "PY_EXIT=$PY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-175 integrate/combat Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel:  combat_gene_scatter_kernel (atomic-scatter α_g),"
echo "               combat_gb_scatter_kernel (per-(g,b) sufficient stats),"
echo "               combat_pooled_var_kernel (pooled σ²_g),"
echo "               combat_z_fill/scatter_kernel (dense Z materialization),"
echo "               combat_gamma_scatter/finalize_kernel (γ/δ² stats),"
echo "               combat_eb_hyperparams_kernel (EB hyperparameters),"
echo "               combat_eb_shrink_kernel × max_iter=2,"
echo "               combat_adjust_kernel (X_adj = (Z-γ*)·σ/√δ²*+α)."
echo "  Algorithm:   ComBat parametric EB batch correction (Johnson 2007)."
echo "  Input:       synthetic dense CSC counts 0-20, seed=42; 4 batches round-robin."
echo ""
echo "  GPU results (singlet-gpu — integrate::combat):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  scanpy CPU baseline (scanpy.pp.combat):"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs scanpy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | scanpy_wall_ms | ratio"
echo "  ------|-------------|----------------|------"

# GPU CSV fields:    scale,n_cells,n_genes,n_batches,max_iter,wall_ms,mem_mb → field 6
# scanpy CSV fields: scale,n_cells,n_genes,n_batches,wall_ms               → field 5
for SCALE in 10k 30k; do
    GPU_MS=$(grep "^${SCALE}," "$GPU_LOG" 2>/dev/null \
             | head -1 | cut -d',' -f6)
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

    printf "  %-5s | %11s | %14s | %s\n" \
        "$SCALE" \
        "${GPU_MS:-N/A}" \
        "${PY_MS:-N/A}" \
        "$RATIO"
done

echo ""
echo "  Feature:    integrate/combat (CYCLE-131)"
echo "  Citation:   Johnson WE et al. (2007) Biostatistics 8:118-127"
echo "  Config:     n_genes=5000, dense, n_batches=4, max_iter=2, seed=42"
echo "  Note:       CPU baseline: scanpy.pp.combat (vectorised numpy EB)"
echo "  Note:       §J.7 predicted class 2-3 (50-300×): vectorized numpy SOTA"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "PY_EXIT=$PY_EXIT"
echo "Date: $(date)"
