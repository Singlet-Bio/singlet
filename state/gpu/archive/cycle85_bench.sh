#!/bin/bash
#SBATCH --job-name=sg_cycle85_bench
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle85_bench_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle85_bench
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 85: DE Wilcoxon + Ttest multi-scale bench vs scanpy CPU ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date

# ─── Build ────────────────────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"

echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1 | tail -10
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit "$CMAKE_EXIT"; }

echo "--- cmake build: bench_de_ttest_perf + bench_de_wilcoxon_perf_c85 ---"
cmake --build "$BUILD_DIR" \
    --target bench_de_ttest_perf bench_de_wilcoxon_perf_c85 \
    -j8 2>&1 | tail -40
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED"; exit "$BUILD_EXIT"; }

# ─── GPU check ────────────────────────────────────────────────────────────────
echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null || true

# ─── Wilcoxon multi-scale bench ───────────────────────────────────────────────
echo ""
echo "=== WILCOXON BENCH (3 scales) ==="
"$BUILD_DIR/bench/bench_de_wilcoxon_perf_c85" 2>&1 | tee /tmp/c85_wilcoxon_${SLURM_JOB_ID}.txt
WILCOXON_EXIT=$?
echo "WILCOXON_EXIT=$WILCOXON_EXIT"

# ─── Ttest multi-scale bench ──────────────────────────────────────────────────
echo ""
echo "=== TTEST BENCH (3 scales) ==="
"$BUILD_DIR/bench/bench_de_ttest_perf" 2>&1 | tee /tmp/c85_ttest_${SLURM_JOB_ID}.txt
TTEST_EXIT=$?
echo "TTEST_EXIT=$TTEST_EXIT"

# ─── Extract key metrics ──────────────────────────────────────────────────────
echo ""
echo "=== KEY METRICS: Wilcoxon ==="
grep -E "BENCH|SUMMARY|SKIP|skipped|small|medium|large" /tmp/c85_wilcoxon_${SLURM_JOB_ID}.txt | head -30 || true

echo ""
echo "=== KEY METRICS: Ttest ==="
grep -E "BENCH|SUMMARY|SKIP|skipped|small|medium|large" /tmp/c85_ttest_${SLURM_JOB_ID}.txt | head -30 || true

# ─── Device memory snapshot ───────────────────────────────────────────────────
echo ""
echo "--- Device memory after runs ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo ""
echo "=== FINAL SUMMARY ==="
echo "BUILD_EXIT=$BUILD_EXIT"
echo "WILCOXON_EXIT=$WILCOXON_EXIT"
echo "TTEST_EXIT=$TTEST_EXIT"
date
