#!/bin/bash
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/cycle86_profile_job.sh
#
# OPTIM-NMF-K50 Phase B — Nsight Systems + Nsight Compute profiling job.
#
# Strategy:
#   1. Build bench_nmf_profile_c86 on the GPU node.
#   2. Run nsys profile at k=20 (baseline) and k=50 (regressed) separately.
#   3. Run ncu --set full on batch_cd_nnls_gpu at k=50 to get:
#        - achieved occupancy
#        - compute vs memory bound
#        - DRAM throughput
#        - register usage / spill
#   4. Run bench_reduce_nmf_c86 for wall-time comparison (with Cycle 86 fix active).
#
# Profiling targets:
#   - nsys: full timeline → shows GEMM vs SpMM vs CD kernel proportions
#   - ncu:  batch_cd_kernel_tpc_dynamic (k=50 CD path) + compute_gram_gpu + rhs

#SBATCH --job-name=nmf-profile-c86
#SBATCH --partition=gpu
#SBATCH --nodelist=g051
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle86_profile_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle86_profile_%j.err

set -uo pipefail

# ---- CUDA environment ----
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_HOME="/usr/local/cuda"
export CUDACXX="/usr/local/cuda/bin/nvcc"

SINGLET_GPU_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
BUILD_DIR=${SINGLET_GPU_DIR}/build_cycle86_profile
TRACES_DIR=${SINGLET_GPU_DIR}/bench/traces
LOGS_DIR=${SINGLET_GPU_DIR}/bench/logs

mkdir -p "${BUILD_DIR}" "${TRACES_DIR}" "${LOGS_DIR}"

echo "=== OPTIM-NMF-K50 Phase B Profiling Job ==="
echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
echo "Date: $(date -Iseconds)"
echo ""

# ---- Step 1: Build the profiling driver ----
echo "--- Step 1: cmake configure + build bench_nmf_profile_c86 ---"
cd "${BUILD_DIR}"

cmake "${SINGLET_GPU_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER="${CUDACXX}" \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DBUILD_TESTING=OFF \
    -DFACTORNET_INCLUDE_DIR=/mnt/home/debruinz/factornet/include \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    > /dev/null 2>&1
echo "CMake configure: done (exit $?)"

# Build only the profiling bench driver (suppress verbose nvcc warnings)
cmake --build "${BUILD_DIR}" --target bench_nmf_profile_c86 -j8 > /dev/null 2>&1
echo "bench_nmf_profile_c86 build: exit $?"

# Also build the c86 verification bench (wall-time with fix)
cmake --build "${BUILD_DIR}" --target bench_reduce_nmf_c86  -j8 > /dev/null 2>&1
echo "bench_reduce_nmf_c86 build: exit $?"

PROFILE_BIN="${BUILD_DIR}/bench/bench_nmf_profile_c86"
C86_BIN="${BUILD_DIR}/bench/bench_reduce_nmf_c86"

echo "Build complete."
echo "  profile binary: ${PROFILE_BIN}"
echo "  c86 bench binary: ${C86_BIN}"
echo ""

# ---- Step 2: Warm run (no profiler) — get verbose per-phase timing ----
echo "--- Step 2: Verbose per-phase timing (no profiler overhead) ---"
echo "--- This uses factornet's built-in GPU timer (gpu_timer.cuh) ---"
echo ""
${PROFILE_BIN} 2>&1
echo ""

# ---- Step 3: nsys profile — k=20 baseline ----
echo "--- Step 3a: nsys profile k=20 (baseline) ---"
NSYS_K20="${TRACES_DIR}/nmf_k20_c86"
nsys profile \
    --output="${NSYS_K20}" \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    ${PROFILE_BIN} \
    2>&1 | grep -E "PROFILE|RESULT|gram|rhs|nnls|norm|loss|BENCH|Time\(%\)|Type|Name|Total" | head -80
echo "nsys k=20 report: ${NSYS_K20}.nsys-rep"
echo ""

# ---- Step 4: nsys profile — k=50 auto (regression) ----
echo "--- Step 3b: nsys profile k=50 auto (regression path) ---"
NSYS_K50="${TRACES_DIR}/nmf_k50_auto_c86"
nsys profile \
    --output="${NSYS_K50}" \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --force-overwrite=true \
    --stats=true \
    --export=sqlite \
    ${PROFILE_BIN} \
    2>&1 | grep -E "PROFILE|RESULT|gram|rhs|nnls|norm|loss|BENCH|Time\(%\)|Type|Name|Total" | head -80
echo "nsys k=50 report: ${NSYS_K50}.nsys-rep"
echo ""

# ---- Step 5: nsys stats — show kernel-level breakdown ----
echo "--- Step 4: nsys stats — top-10 kernels by time (k=20 vs k=50) ---"
echo ""
echo "=== k=20 kernel breakdown ==="
nsys stats "${NSYS_K20}.nsys-rep" \
    --report=cuda_gpu_kern_sum \
    --format=table 2>&1 | head -25 || echo "nsys stats unavailable or report format changed"

echo ""
echo "=== k=50 kernel breakdown ==="
nsys stats "${NSYS_K50}.nsys-rep" \
    --report=cuda_gpu_kern_sum \
    --format=table 2>&1 | head -25 || echo "nsys stats unavailable"

# ---- Step 6: ncu — batch_cd_kernel_tpc_dynamic at k=50 ----
echo ""
echo "--- Step 5: ncu --set full on k=50 CD kernel ---"
echo "Target kernel: batch_cd_kernel_tpc_dynamic"
NCU_K50="${TRACES_DIR}/nmf_k50_cd_ncu_c86"

ncu \
    --set full \
    --kernel-name "batch_cd_kernel_tpc_dynamic" \
    --launch-count 3 \
    --output "${NCU_K50}" \
    --force-overwrite \
    --csv \
    ${PROFILE_BIN} \
    2>&1 | grep -E "batch_cd|occupancy|throughput|bound|register|spill|Achieved|Compute|Memory|DRAM|sm__|achieved" | head -40
echo "ncu report: ${NCU_K50}.ncu-rep"

# ---- Step 7: ncu on compute_gram (cuBLAS SGEMM) at k=50 ----
echo ""
echo "--- Step 6: ncu on cuBLAS SGEMM (gram computation) at k=50 ---"
NCU_GRAM="${TRACES_DIR}/nmf_k50_gram_ncu_c86"
ncu \
    --set roofline \
    --kernel-name-base function \
    --kernel-name "ampere_sgemm\|cgemm\|gemm" \
    --launch-count 5 \
    --output "${NCU_GRAM}" \
    --force-overwrite \
    --csv \
    ${PROFILE_BIN} \
    2>&1 | grep -E "gemm|sgemm|occupancy|throughput|Compute|Memory|DRAM|arithmetic" | head -40
echo "ncu gram report: ${NCU_GRAM}.ncu-rep"

# ---- Step 8: Wall-time comparison (Cycle 86 fix active) ----
echo ""
echo "--- Step 7: Wall-time bench with Cycle 86 MU-forcing fix ---"
echo "Expected: k=20 ~400ms, k=50 <1500ms (MU path), k=100 Cholesky"
${C86_BIN} 2>&1
echo ""

# ---- Step 9: Summary of file paths ----
echo ""
echo "=== OUTPUT FILES ==="
echo "nsys k=20 report:        ${NSYS_K20}.nsys-rep"
echo "nsys k=50 report:        ${NSYS_K50}.nsys-rep"
echo "ncu  k=50 CD report:     ${NCU_K50}.ncu-rep"
echo "ncu  k=50 gram report:   ${NCU_GRAM}.ncu-rep"
echo "Log: ${LOGS_DIR}/cycle86_profile_${SLURM_JOB_ID}.log"
echo ""
echo "=== Job complete ==="
