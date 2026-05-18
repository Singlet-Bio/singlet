#!/bin/bash
#SBATCH --job-name=sg_c86_nmf_mu
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle86_nmf_mu_default_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle86_verify
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 86 / OPTIM-NMF-K50: adapter MU-default fix verification ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date

# ── Build ──────────────────────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"

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
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit "$CMAKE_EXIT"; }

echo "--- cmake build: reduce_nmf_correctness + bench_reduce_nmf_c86 + DE tests ---"
cmake --build "$BUILD_DIR" \
    --target reduce_nmf_correctness bench_reduce_nmf_c86 \
            de_wilcoxon_correctness de_ttest_correctness \
    -j8 2>&1 | tail -40
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED"; exit "$BUILD_EXIT"; }

# ── GPU info ───────────────────────────────────────────────────────────────────
echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ── NMF correctness ctest ──────────────────────────────────────────────────────
echo ""
echo "=== NMF ctest (all NMF correctness tests) ==="
ctest --test-dir "$BUILD_DIR" -R "nmf|NMF|Nmf" \
      --output-on-failure -V 2>&1 | tee /tmp/c86_nmf_ctest_${SLURM_JOB_ID}.txt
NMF_CTEST_EXIT=$?
echo "NMF_CTEST_EXIT=$NMF_CTEST_EXIT"

NMF_PASS=$(grep -c "PASSED" /tmp/c86_nmf_ctest_${SLURM_JOB_ID}.txt 2>/dev/null || echo 0)
NMF_FAIL=$(grep -c "FAILED" /tmp/c86_nmf_ctest_${SLURM_JOB_ID}.txt 2>/dev/null || echo 0)
echo "NMF ctest: PASS=$NMF_PASS  FAIL=$NMF_FAIL"

# ── NMF multi-k bench ─────────────────────────────────────────────────────────
echo ""
echo "=== NMF multi-k bench (k=10,20,50,100) ==="
"$BUILD_DIR/bench/bench_reduce_nmf_c86" 2>&1 | tee /tmp/c86_nmf_bench_${SLURM_JOB_ID}.txt
BENCH_EXIT=$?
echo "BENCH_EXIT=$BENCH_EXIT"

echo ""
echo "=== KEY BENCH RESULTS ==="
grep -E "BENCH|SUMMARY|SKIP|==" /tmp/c86_nmf_bench_${SLURM_JOB_ID}.txt | head -20 || true

# ── DE regression ctest ───────────────────────────────────────────────────────
echo ""
echo "=== DE regression ctest ==="
ctest --test-dir "$BUILD_DIR" \
    -R "Wilcoxon.*TinyPlanted|Ttest.*TinyPlanted|Test83_RealDataSized_PostNormalize_NoCrash|Wilcoxon_GSM4037629_RealDataPlanted|Ttest_GSM4037629_RealDataPlanted" \
    --output-on-failure -V 2>&1 | tee /tmp/c86_de_ctest_${SLURM_JOB_ID}.txt
DE_CTEST_EXIT=$?
echo "DE_CTEST_EXIT=$DE_CTEST_EXIT"

DE_PASS=$(grep -c "PASSED" /tmp/c86_de_ctest_${SLURM_JOB_ID}.txt 2>/dev/null || echo 0)
DE_FAIL=$(grep -c "FAILED" /tmp/c86_de_ctest_${SLURM_JOB_ID}.txt 2>/dev/null || echo 0)
echo "DE ctest: PASS=$DE_PASS  FAIL=$DE_FAIL"

echo ""
echo "=== DE regression key metrics ==="
grep -E "Jaccard|Spearman|PASSED|FAILED|PASS|FAIL|NaN|CUDA error" \
    /tmp/c86_de_ctest_${SLURM_JOB_ID}.txt | head -30 || true

# ── Device memory snapshot ────────────────────────────────────────────────────
echo ""
echo "--- Device memory after all runs ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo ""
echo "=== FINAL SUMMARY ==="
echo "CMAKE_EXIT=$CMAKE_EXIT"
echo "BUILD_EXIT=$BUILD_EXIT"
echo "NMF_CTEST_EXIT=$NMF_CTEST_EXIT  NMF PASS=$NMF_PASS FAIL=$NMF_FAIL"
echo "BENCH_EXIT=$BENCH_EXIT"
echo "DE_CTEST_EXIT=$DE_CTEST_EXIT  DE PASS=$DE_PASS FAIL=$DE_FAIL"
date
