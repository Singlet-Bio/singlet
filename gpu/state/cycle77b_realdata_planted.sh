#!/bin/bash
#SBATCH --job-name=sg_cycle77b_realdata
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle77b_realdata_planted_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu/build_cycle77b_verify
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 77b RealDataPlanted pval-fix retest ==="
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
    2>&1 | tail -10
CMAKE_EXIT=$?
echo "CMAKE_EXIT=$CMAKE_EXIT"
[ "$CMAKE_EXIT" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit $CMAKE_EXIT; }

echo "--- cmake build (target de_wilcoxon_correctness) ---"
cmake --build "$BUILD_DIR" --target de_wilcoxon_correctness -j8 2>&1 | tail -20
BUILD_EXIT=$?
echo "BUILD_EXIT=$BUILD_EXIT"
[ "$BUILD_EXIT" -ne 0 ] && { echo "BUILD FAILED"; exit $BUILD_EXIT; }

echo ""
echo "--- ctest: Wilcoxon_GSM4037629_RealDataPlanted (primary target) ---"
ctest --test-dir "$BUILD_DIR" -R "Wilcoxon_GSM4037629_RealDataPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/realplanted_${SLURM_JOB_ID}.txt
REALPLANTED_EXIT=$?
echo "REALPLANTED_EXIT=$REALPLANTED_EXIT"

echo ""
echo "--- ctest: Wilcoxon.*TinyPlanted (no-regression) ---"
ctest --test-dir "$BUILD_DIR" -R "Wilcoxon.*TinyPlanted" \
      --output-on-failure -V 2>&1 | tee /tmp/tinyplanted_${SLURM_JOB_ID}.txt
TINY_EXIT=$?
echo "TINY_EXIT=$TINY_EXIT"

echo ""
echo "--- ctest: Test83_RealDataSized_PostNormalize_NoCrash (no-regression) ---"
ctest --test-dir "$BUILD_DIR" -R "Test83_RealDataSized_PostNormalize_NoCrash" \
      --output-on-failure -V 2>&1 | tee /tmp/test83_${SLURM_JOB_ID}.txt
TEST83_EXIT=$?
echo "TEST83_EXIT=$TEST83_EXIT"

echo ""
echo "=== KEY METRICS FROM RealDataPlanted ==="
grep -E "Jaccard|Spearman|LFC|PvalRank|registry|PASSED|FAILED|intersection" \
    /tmp/realplanted_${SLURM_JOB_ID}.txt | head -60 || true

echo ""
echo "=== KEY METRICS FROM TinyPlanted ==="
grep -E "PASSED|FAILED|PASS|FAIL" \
    /tmp/tinyplanted_${SLURM_JOB_ID}.txt | head -10 || true

echo ""
echo "=== TEST83 VERDICT ==="
grep -E "PASSED|FAILED|PASS|FAIL|CUDA error" \
    /tmp/test83_${SLURM_JOB_ID}.txt | head -5 || true

echo ""
echo "=== SUMMARY ==="
echo "BUILD_EXIT=$BUILD_EXIT  REALPLANTED_EXIT=$REALPLANTED_EXIT  TINY_EXIT=$TINY_EXIT  TEST83_EXIT=$TEST83_EXIT"
date
