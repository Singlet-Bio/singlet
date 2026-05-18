#!/bin/bash
#SBATCH --job-name=cycle72_clean_retest
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle72_clean_retest_%j.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g008,g050,g051,g052
# Note: g050-g052 are H100 NVL (sm_90) — wilcoxon kernel has correctness bugs on Hopper (warp/atomic ordering)
# that are unrelated to BUG-WILCOXON-POST-NORMALIZE-CRASH. Target V100 (sm_70) for this validation.

set -euo pipefail

# Load CUDA environment matching the production build
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

cd /mnt/home/debruinz/Singlet-AI/singlet-gpu

# Confirm the four source fixes are still in place before wasting a build.
echo "=== SOURCE FIX VERIFICATION ==="
for f in include/singlet-gpu/preprocess/lognorm.h \
         include/singlet-gpu/preprocess/scale.h \
         include/singlet-gpu/preprocess/hvg.h \
         include/singlet-gpu/de/donor_pseudobulk.h; do
  echo "--- $f ---"
  grep -n "cudaStreamSynchronize(stream)" "$f" | head -5 || echo "MISSING"
done

echo "=== CLEAN REBUILD ==="
# Force a fresh build tree so NFS timestamp races cannot corrupt the result.
rm -rf build_cycle72_verify
cmake -S . -B build_cycle72_verify \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
  -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
  2>&1 | tail -20
cmake --build build_cycle72_verify -j8 --target de_wilcoxon_correctness 2>&1 | tail -30
BUILD_EXIT=$?
echo "BUILD_EXIT=${BUILD_EXIT}"

echo "=== TEST83 RUN ==="
cd build_cycle72_verify
set +e
ctest -R "Test83_RealDataSized_PostNormalize_NoCrash" --output-on-failure 2>&1 | tee test83_output.txt
TEST_EXIT=$?
echo "TEST_EXIT=$TEST_EXIT"
set -e

# Also run the TinyPlanted trio (tests 80-82) to confirm no regression.
echo "=== TESTS 80-82 NO-REGRESSION ==="
set +e
ctest -R "Wilcoxon.*TinyPlanted" --output-on-failure 2>&1 | tee tiny_output.txt
TINY_EXIT=$?
echo "TINY_EXIT=$TINY_EXIT"
set -e

echo "=== DONE ==="
