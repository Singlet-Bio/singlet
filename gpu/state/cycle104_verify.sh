#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 104 verify — compile the public umbrella header + cpp_minimal smoke test.

#SBATCH --job-name=sg_c104_verify
#SBATCH --partition=gpu
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle104_verify_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle104_pybind
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu

echo "=== Cycle 104 verify: public umbrella header ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "g++: $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

rm -rf "$BUILD_DIR"

echo ""
echo "--- standalone nvcc compile of examples/cpp_minimal/main.cpp ---"
# The umbrella pulls in pz_device_loader.h which uses <<< >>> kernel launch
# syntax inline. Every downstream consumer compiles with nvcc; so do we.
nvcc -std=c++20 -O2 -DFACTORNET_HAS_GPU=1 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -x cu \
    -c "$SRC_DIR/examples/cpp_minimal/main.cpp" \
    -o "/dev/shm/cpp_minimal_${SLURM_JOB_ID}.o" 2>&1 | tail -40
COMPILE_EXIT=${PIPESTATUS[0]}
echo "COMPILE_EXIT=${COMPILE_EXIT}"

if [ "${COMPILE_EXIT}" -eq 0 ]; then
    echo "=== Public umbrella compile: PASS ==="
    rm -f "/dev/shm/cpp_minimal_${SLURM_JOB_ID}.o"
else
    echo "=== Public umbrella compile: FAIL ==="
fi

echo ""
echo "--- header self-containment check (nvcc) ---"
# Compile a translation unit that only includes the umbrella + uses
# version_major(). Catches missing transitive includes.
cat > /dev/shm/sg_self_${SLURM_JOB_ID}.cu <<'CU_EOF'
#include <singlet-gpu/singlet_gpu.hpp>
int main() { return singlet_gpu::version_major(); }
CU_EOF
nvcc -std=c++20 -O2 -DFACTORNET_HAS_GPU=1 \
    -ccbin "${CXX}" \
    -gencode arch=compute_70,code=sm_70 \
    --expt-relaxed-constexpr \
    -I "$SRC_DIR/include" \
    -I /mnt/home/debruinz/factornet/include \
    -I /opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    -c "/dev/shm/sg_self_${SLURM_JOB_ID}.cu" \
    -o "/dev/shm/sg_self_${SLURM_JOB_ID}.o" 2>&1 | tail -25
SELF_EXIT=${PIPESTATUS[0]}
echo "SELF_EXIT=${SELF_EXIT}"
rm -f /dev/shm/sg_self_${SLURM_JOB_ID}.{cu,o}

echo ""
echo "=== FINAL EXIT CODES ==="
echo "COMPILE_EXIT=${COMPILE_EXIT}"
echo "SELF_EXIT=${SELF_EXIT}"
date
