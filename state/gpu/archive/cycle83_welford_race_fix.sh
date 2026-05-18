#!/bin/bash
#SBATCH --job-name=sg_cycle83_welford
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle83_welford_race_fix_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle83_verify
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 83: Welford race fix (sum+sum_sq two-pass) ==="
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
    2>&1 | tail -5
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit $CMAKE_EXIT; }

echo "--- cmake build (targets: de_ttest_correctness, de_wilcoxon_correctness) ---"
cmake --build "$BUILD_DIR" --target de_ttest_correctness de_wilcoxon_correctness -j8 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED"; exit $BUILD_EXIT; }

echo ""
echo "--- ctest: Ttest TinyPlanted (all 3 metrics) ---"
ctest --test-dir "$BUILD_DIR" \
      -R "Ttest_TinyPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/c83_tinyplanted_${SLURM_JOB_ID}.txt
TTEST_TINY_EXIT=$?
echo "TTEST_TINY_EXIT=$TTEST_TINY_EXIT"

echo ""
echo "--- ctest: Ttest GSM4037629 RealData ---"
ctest --test-dir "$BUILD_DIR" \
      -R "Ttest_GSM4037629_RealData" \
      --output-on-failure -V 2>&1 | tee /tmp/c83_realdata_${SLURM_JOB_ID}.txt
TTEST_REAL_EXIT=$?
echo "TTEST_REAL_EXIT=$TTEST_REAL_EXIT"

echo ""
echo "--- ctest: Wilcoxon TinyPlanted (regression) ---"
ctest --test-dir "$BUILD_DIR" \
      -R "Wilcoxon_TinyPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/c83_wilcoxon_${SLURM_JOB_ID}.txt
WILCOXON_EXIT=$?
echo "WILCOXON_EXIT=$WILCOXON_EXIT"

echo ""
echo "--- ctest: Test83 + RealDataPlanted (regression) ---"
ctest --test-dir "$BUILD_DIR" \
      -R "Test83_RealDataSized_PostNormalize_NoCrash|Wilcoxon_GSM4037629_RealDataPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/c83_regression_${SLURM_JOB_ID}.txt
REGRESSION_EXIT=$?
echo "REGRESSION_EXIT=$REGRESSION_EXIT"

echo ""
echo "=== KEY METRICS: TinyPlanted ==="
grep -E "Jaccard|LFCSpearman|PvalRank|Spearman|PASSED|FAILED|PASS|FAIL" \
    /tmp/c83_tinyplanted_${SLURM_JOB_ID}.txt | head -30 || true

echo ""
echo "=== KEY METRICS: RealData ==="
grep -E "Jaccard|PASSED|FAILED|PASS|FAIL" \
    /tmp/c83_realdata_${SLURM_JOB_ID}.txt | head -20 || true

echo ""
echo "=== KEY METRICS: Wilcoxon regression ==="
grep -E "PASSED|FAILED|PASS|FAIL" \
    /tmp/c83_wilcoxon_${SLURM_JOB_ID}.txt | head -10 || true

echo ""
echo "=== KEY METRICS: Test83 + RealDataPlanted ==="
grep -E "PASSED|FAILED|PASS|FAIL" \
    /tmp/c83_regression_${SLURM_JOB_ID}.txt | head -10 || true

echo ""
echo "=== SUMMARY ==="
echo "BUILD_EXIT=$BUILD_EXIT"
echo "TTEST_TINY_EXIT=$TTEST_TINY_EXIT"
echo "TTEST_REAL_EXIT=$TTEST_REAL_EXIT"
echo "WILCOXON_EXIT=$WILCOXON_EXIT"
echo "REGRESSION_EXIT=$REGRESSION_EXIT"
date
