#!/bin/bash
#SBATCH --job-name=sg55c_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla_v100s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle55c_fix_test_%j.log

echo "=== NODE: $(hostname) === GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1) ==="
echo "=== START: $(date) ==="

source /opt/rh/gcc-toolset-13/enable
export PATH=/usr/local/cuda-12.6/bin:$PATH
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc

echo "=== COMPILER ==="
gcc --version | head -1
nvcc --version | head -1

EIGEN3_DIR=/opt/gvsu/clipper/2024.12/spack/apps/linux-rhel9-cascadelake/gcc-13.3.1/eigen-3.4.0-cjmrfcdntigsqf5uh3q6xdbm6twlooby/share/eigen3/cmake

BUILD_DIR=/dev/shm/singletgpu_fix55c_${SLURM_JOB_ID}
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
if [ ${CMAKE_EXIT} -ne 0 ]; then echo "CMAKE FAILED"; exit 1; fi

echo "=== BUILD ==="
make -j16 preprocess_hvg_correctness 2>&1
BUILD_EXIT=$?
echo "BUILD exit: ${BUILD_EXIT}"
if [ ${BUILD_EXIT} -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi

# Clear cached refs to force fresh reference computation
rm -f /tmp/singlet_gpu_hvg_refs_tmp/hvg_ref_tiny.npz
rm -f /tmp/singlet_gpu_hvg_refs_tmp/hvg_ref_10k.npz

echo "=== HVG TESTS (all 8) ==="
cd ${BUILD_DIR}
ctest --output-on-failure -R 'HvgTest' -V --timeout 600 2>&1
HVG_EXIT=$?
echo "HVG tests exit: ${HVG_EXIT}"

echo "=== END: $(date) ==="
rm -rf ${BUILD_DIR}
