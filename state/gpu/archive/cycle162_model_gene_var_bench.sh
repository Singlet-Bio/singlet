#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# CYCLE-162 Phase E — preprocess/model_gene_var medium-scale bench vs scanpy CPU.
#
# GPU kernel: port of scran::modelGeneVarByPoisson (Lun-McCarthy-Marioni 2016).
# Scales: 10k cells × 5k genes (density 5%) and 30k cells × 5k genes.
# Algorithm: O(nnz) atomic scatter + cub::DeviceRadixSort top-N — NOT at risk
# of the diffmap/dpt dense-n×n scaling bug (§J.6 audit, CYCLE-160).
#
# §J.2 node routing: --exclude=g001,g002,g005 (not --nodelist=).
# Target: any non-excluded GPU node (g003/g004/g050/g051/g052).
#
# Scanpy baselines:
#   flavor='pearson_residuals' — closest algorithm (Poisson-null, Lause 2021)
#   flavor='seurat_v3'         — most popular HVG flavor (Stuart 2019)
# Two timing rows per scale; same input seed=42 as GPU bench.
#
# 30-min walltime. Output: state/cycle162_model_gene_var_bench_%j.log.
# Build: only bench_preprocess_model_gene_var_perf target.

#SBATCH --job-name=mgv_bench_c162
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle162_model_gene_var_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle162_mgv
GPU_LOG=/tmp/cycle162_gpu_${SLURM_JOB_ID}.csv
PY_LOG=/tmp/cycle162_scanpy_${SLURM_JOB_ID}.csv

echo "=== CYCLE-162: preprocess/model_gene_var Phase E bench ==="
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
echo "--- cmake build: bench_preprocess_model_gene_var_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_preprocess_model_gene_var_perf \
    -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED — stopping."; exit 1; }

# --- GPU bench ---------------------------------------------------------------
echo ""
echo "=== GPU BENCH: preprocess/model_gene_var (singlet-gpu, 10k+30k cells) ==="
"$BUILD_DIR/bench/bench_preprocess_model_gene_var_perf" 2>&1 | tee "$GPU_LOG"
GPU_EXIT=$?
echo "GPU_EXIT=$GPU_EXIT"

# --- scanpy CPU baselines ----------------------------------------------------
echo ""
echo "=== SCANPY CPU BASELINE: sc.pp.highly_variable_genes (pearson_residuals + seurat_v3) ==="
python3 "$SRC_DIR/bench/refs/model_gene_var_ref.py" 2>&1 | tee "$PY_LOG"
SCANPY_EXIT=$?
echo "SCANPY_EXIT=$SCANPY_EXIT"

# --- SUMMARY -----------------------------------------------------------------
echo ""
echo "==================================================================="
echo "  SUMMARY — CYCLE-162 preprocess/model_gene_var Phase E bench"
echo "==================================================================="
echo ""
echo "  GPU kernel: scran::modelGeneVarByPoisson port (CYCLE-127)"
echo "  Algorithm:  O(nnz) scatter + cub::DeviceRadixSort top-N HVG"
echo "  §J.6 status: NOT at risk (no dense n×n materialization)"
echo ""
echo "  GPU results (singlet-gpu — model_gene_var, n_top=2000):"
grep -E "^(10k|30k|scale)" "$GPU_LOG" 2>/dev/null | head -10 \
    || echo "  (no CSV rows found)"

echo ""
echo "  scanpy CPU baselines:"
grep -E "^(10k|30k|scale)" "$PY_LOG" 2>/dev/null | head -20 \
    || echo "  (no CSV rows found)"

echo ""
echo "  Side-by-side ratio table (GPU wall_ms vs scanpy wall_ms):"
echo ""
echo "  scale | GPU_wall_ms | scanpy_pearson_ms | scanpy_seurat_v3_ms | ratio_pearson | ratio_seurat_v3"
echo "  ------|-------------|-------------------|---------------------|---------------|----------------"

# GPU CSV fields:    scale,n_cells,n_genes,density,n_top,wall_ms,mem_mb  → field 6
# scanpy CSV fields: scale,n_cells,n_genes,density,n_top,flavor,wall_ms  → field 7
for SCALE in 10k 30k; do
    GPU_MS=$(grep "^${SCALE}," "$GPU_LOG" 2>/dev/null | head -1 | cut -d',' -f6)
    SC_PR=$(grep "^${SCALE}," "$PY_LOG" 2>/dev/null | grep "pearson_residuals" | head -1 | cut -d',' -f7)
    SC_SV=$(grep "^${SCALE}," "$PY_LOG" 2>/dev/null | grep "seurat_v3"         | head -1 | cut -d',' -f7)

    RATIO_PR="N/A"
    RATIO_SV="N/A"
    if [ -n "${GPU_MS:-}" ] && [ -n "${SC_PR:-}" ]; then
        RATIO_PR=$(python3 -c "
g=float('${GPU_MS}'); s=float('${SC_PR}')
if g>0 and s>0:
    r=s/g
    print(f'{r:.2f}x (GPU faster)' if r>1 else f'{1/r:.2f}x (GPU slower)')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
    fi
    if [ -n "${GPU_MS:-}" ] && [ -n "${SC_SV:-}" ]; then
        RATIO_SV=$(python3 -c "
g=float('${GPU_MS}'); s=float('${SC_SV}')
if g>0 and s>0:
    r=s/g
    print(f'{r:.2f}x (GPU faster)' if r>1 else f'{1/r:.2f}x (GPU slower)')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")
    fi

    printf "  %-5s | %11s | %17s | %19s | %-13s | %s\n" \
        "$SCALE" \
        "${GPU_MS:-N/A}" \
        "${SC_PR:-N/A}" \
        "${SC_SV:-N/A}" \
        "$RATIO_PR" \
        "$RATIO_SV"
done

echo ""
echo "  Feature:   preprocess/model_gene_var (CYCLE-127)"
echo "  Citation:  Lun ATL, McCarthy DJ, Marioni JC (2016) F1000Research 5:2122"
echo "  Config:    n_top=2000, min_mean=0.0, density=5%, seed=42"
echo "  Note:      GPU covers full model_gene_var() path (passes 1-4)"
echo "  Note:      scanpy pearl_residuals = closest algorithm to the GPU kernel"
echo "  Note:      scanpy seurat_v3 = most popular HVG flavor (different algorithm)"
echo "==================================================================="
echo ""
echo "BUILD_EXIT=$BUILD_EXIT"
echo "GPU_EXIT=$GPU_EXIT"
echo "SCANPY_EXIT=$SCANPY_EXIT"
echo "Date: $(date)"
