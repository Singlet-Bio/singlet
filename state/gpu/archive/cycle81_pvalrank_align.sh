#!/bin/bash
#SBATCH --job-name=sg_c81_pvalrank_align
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle81_pvalrank_align_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle81_verify
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 81: gene-aligned PvalueRankSpearman fix ==="
echo "Job: $SLURM_JOB_ID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
date

rm -rf "$BUILD_DIR"

echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1 | tail -20
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit $CMAKE_EXIT; }

echo "--- cmake build: de_ttest_correctness ---"
cmake --build "$BUILD_DIR" --target de_ttest_correctness -j8 2>&1 | tail -20
BUILD_TTEST=$?
echo "BUILD_TTEST=$BUILD_TTEST"

echo "--- cmake build: de_wilcoxon_correctness ---"
cmake --build "$BUILD_DIR" --target de_wilcoxon_correctness -j8 2>&1 | tail -20
BUILD_WIL=$?
echo "BUILD_WIL=$BUILD_WIL"

BUILD_EXIT=$(( BUILD_TTEST || BUILD_WIL ))
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED"; exit $BUILD_EXIT; }

# --- PRIMARY: Ttest TinyPlanted ---
# Expect: PvalueRankSpearman jumps from 0.0 toward 1.0 for clusters with Jaccard=1.0.
echo ""
echo "--- ctest: Ttest.*TinyPlanted ---"
ctest --test-dir "$BUILD_DIR" -R "Ttest.*TinyPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/c81pa_ttest_${SLURM_JOB_ID}.txt
TTEST_TINY_EXIT=$?
echo "TTEST_TINY_EXIT=$TTEST_TINY_EXIT"

# --- REGRESSION: Wilcoxon TinyPlanted ---
echo ""
echo "--- ctest: Wilcoxon.*TinyPlanted ---"
ctest --test-dir "$BUILD_DIR" -R "Wilcoxon.*TinyPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/c81pa_wil_${SLURM_JOB_ID}.txt
WIL_TINY_EXIT=$?
echo "WIL_TINY_EXIT=$WIL_TINY_EXIT"

# --- REGRESSION: Test83 ---
echo ""
echo "--- ctest: Test83_RealDataSized_PostNormalize_NoCrash ---"
ctest --test-dir "$BUILD_DIR" -R "Test83_RealDataSized_PostNormalize_NoCrash" \
      --output-on-failure -V 2>&1 | tee /tmp/c81pa_test83_${SLURM_JOB_ID}.txt
TEST83_EXIT=$?
echo "TEST83_EXIT=$TEST83_EXIT"

# --- REGRESSION: RealDataPlanted (wilcoxon) ---
echo ""
echo "--- ctest: Wilcoxon_GSM4037629_RealDataPlanted ---"
ctest --test-dir "$BUILD_DIR" -R "Wilcoxon_GSM4037629_RealDataPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/c81pa_rdp_${SLURM_JOB_ID}.txt
RDP_EXIT=$?
echo "RDP_EXIT=$RDP_EXIT"

echo ""
echo "=== KEY METRICS: Ttest TinyPlanted ==="
grep -E "Jaccard|Spearman|rho|PASSED|FAILED|PASS|FAIL|intersection" /tmp/c81pa_ttest_${SLURM_JOB_ID}.txt | head -50 || true

echo ""
echo "=== KEY METRICS: Wilcoxon TinyPlanted ==="
grep -E "Jaccard|Spearman|rho|PASSED|FAILED|PASS|FAIL|intersection" /tmp/c81pa_wil_${SLURM_JOB_ID}.txt | head -40 || true

echo ""
echo "=== REGRESSION Test83 verdict ==="
grep -E "PASSED|FAILED|PASS|FAIL|NaN|CUDA error" /tmp/c81pa_test83_${SLURM_JOB_ID}.txt | head -10 || true

echo ""
echo "=== REGRESSION RealDataPlanted verdict ==="
grep -E "Jaccard|Spearman|PASSED|FAILED|PASS|FAIL" /tmp/c81pa_rdp_${SLURM_JOB_ID}.txt | head -20 || true

echo ""
echo "=== FINAL SUMMARY ==="
echo "BUILD_EXIT=$BUILD_EXIT"
echo "TTEST_TINY_EXIT=$TTEST_TINY_EXIT  WIL_TINY_EXIT=$WIL_TINY_EXIT"
echo "TEST83_EXIT=$TEST83_EXIT  RDP_EXIT=$RDP_EXIT"
date
