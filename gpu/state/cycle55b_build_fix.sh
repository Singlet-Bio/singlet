#!/bin/bash
#SBATCH --job-name=sg55b_build
#SBATCH --partition=gpu
#SBATCH --nodelist=g001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle55b_build_%j.log

echo "=== NODE: $(hostname) === GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1) ==="
echo "=== START: $(date) ==="

# Enable gcc-toolset-13 and CUDA 12.6
source /opt/rh/gcc-toolset-13/enable
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc

# Confirm toolchain
echo "=== COMPILER ==="
gcc --version | head -1
nvcc --version | head -1

# Eigen3 path — gcc-13.3.1 build matching our toolchain
EIGEN3_DIR=/opt/gvsu/clipper/2024.12/spack/apps/linux-rhel9-cascadelake/gcc-13.3.1/eigen-3.4.0-cjmrfcdntigsqf5uh3q6xdbm6twlooby/share/eigen3/cmake

# Fresh build directory in /dev/shm
BUILD_DIR=/dev/shm/singletgpu_build55b_fix_${SLURM_JOB_ID}
mkdir -p ${BUILD_DIR}

echo "=== CMAKE CONFIGURE ==="
cd ${BUILD_DIR}
cmake /mnt/home/debruinz/Singlet-AI/singlet-gpu \
    -DFACTORNET_ROOT=/mnt/home/debruinz/factornet \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DEigen3_DIR=${EIGEN3_DIR}
CMAKE_EXIT=$?
echo "CMAKE exit: ${CMAKE_EXIT}"

if [ ${CMAKE_EXIT} -ne 0 ]; then
    echo "CMAKE FAILED — aborting"
    exit 1
fi

echo "=== BUILD ==="
make -j16 2>&1
BUILD_EXIT=$?
echo "=== BUILD DONE: ${BUILD_EXIT} ==="

if [ ${BUILD_EXIT} -ne 0 ]; then
    echo "BUILD FAILED — aborting ctest"
    # Show last 40 lines of make output for diagnosis
    echo "Last build errors:"
    make -j16 2>&1 | grep -E "error:|collect2" | head -20
    exit 1
fi

echo "=== FULL CTEST (263 tests) ==="
cd ${BUILD_DIR}
# Run with output on failure, parallel 4, timeout 300s per test
ctest --output-on-failure -j4 --timeout 300 2>&1 | tee /tmp/ctest_full_${SLURM_JOB_ID}.log
CTEST_EXIT=$?

echo "=== CTEST PASS/FAIL SUMMARY ==="
grep -E "^\s+[0-9]+/[0-9]+\s+Test|tests passed|tests failed|FAILED:" /tmp/ctest_full_${SLURM_JOB_ID}.log | tail -30
echo "=== CTEST EXIT: ${CTEST_EXIT} ==="

# List all failed tests
echo "=== FAILED TESTS ==="
grep "FAILED" /tmp/ctest_full_${SLURM_JOB_ID}.log | grep -v "^--" || echo "None"

echo "=== END: $(date) ==="

# Cleanup
rm -rf ${BUILD_DIR}
