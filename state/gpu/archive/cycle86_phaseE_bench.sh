#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/state/cycle86_phaseE_bench.sh
#
# Cycle 86 Phase E — ONE job, two benchmarks:
#   A. NMF post-fix adapter bench (bench_nmf_phaseE_c86)
#      k∈{10,20,50,100} × {small synthetic, medium GSM4037629}
#      vs sklearn NMF CPU (refs/nmf_sklearn_ref.py)
#   B. QC metrics bench (bench_qc_perf)
#      {small synthetic, medium GSM4037629}
#      vs scanpy.pp.calculate_qc_metrics CPU (refs/qc_scanpy_ref.py)
#
# Requested: H100 NVL on g051 (gpu partition, gres=gpu:h100nvl:1).
# Fallback: any GPU node with ≥40GB VRAM.

#SBATCH --job-name=sg_c86_phaseE
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodelist=g051
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle86_phaseE_%j.log

set -uo pipefail

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc

BUILD_DIR=/mnt/projects/debruinz_project/singlet_gpu_builds/build_cycle86_phaseE
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
mkdir -p /mnt/projects/debruinz_project/singlet_gpu_builds/
REFS_DIR="${SRC_DIR}/bench/refs"

echo "=== Cycle 86 Phase E: NMF adapter rebench + QC bench vs scanpy ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
date

# ─── Build ────────────────────────────────────────────────────────────────────
rm -rf "$BUILD_DIR"

echo ""
echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DEIGEN_INCLUDE_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3 \
    2>&1 | tail -15
CMAKE_EXIT=$?
echo "CMAKE_EXIT=${CMAKE_EXIT}"
[ "${CMAKE_EXIT}" -ne 0 ] && { echo "CMAKE CONFIGURE FAILED"; exit "${CMAKE_EXIT}"; }

echo ""
echo "--- cmake build: bench_nmf_phaseE_c86 + bench_qc_perf ---"
cmake --build "$BUILD_DIR" \
    --target bench_nmf_phaseE_c86 bench_qc_perf \
    -j12 2>&1 | tail -50
BUILD_EXIT=$?
echo "BUILD_EXIT=${BUILD_EXIT}"
[ "${BUILD_EXIT}" -ne 0 ] && { echo "BUILD FAILED"; exit "${BUILD_EXIT}"; }

# ─── GPU info ─────────────────────────────────────────────────────────────────
echo ""
echo "--- GPU info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ─── Benchmark A: sklearn NMF CPU reference ───────────────────────────────────
echo ""
echo "=== BENCHMARK A — sklearn NMF CPU reference ==="
echo "Running refs/nmf_sklearn_ref.py --scale all ..."

SKLEARN_OUT=/dev/shm/c86_sklearn_nmf_${SLURM_JOB_ID}.txt
python3 "${REFS_DIR}/nmf_sklearn_ref.py" --scale all \
    2>/dev/shm/c86_sklearn_nmf_stderr_${SLURM_JOB_ID}.txt \
    | tee "${SKLEARN_OUT}"
SKLEARN_EXIT=$?
echo "SKLEARN_EXIT=${SKLEARN_EXIT}"

echo ""
echo "--- sklearn stderr (progress) ---"
tail -30 /dev/shm/c86_sklearn_nmf_stderr_${SLURM_JOB_ID}.txt || true

# Extract sklearn ms values for the summary table.
grep "^SKLEARN_WALL_MS=" "${SKLEARN_OUT}" || true

# ─── Benchmark A: GPU NMF adapter ─────────────────────────────────────────────
echo ""
echo "=== BENCHMARK A — GPU NMF adapter (bench_nmf_phaseE_c86) ==="
"${BUILD_DIR}/bench/bench_nmf_phaseE_c86" \
    2>&1 | tee /dev/shm/c86_nmf_phaseE_${SLURM_JOB_ID}.txt
NMF_EXIT=$?
echo "NMF_EXIT=${NMF_EXIT}"

# ─── Benchmark B: scanpy QC CPU reference ─────────────────────────────────────
echo ""
echo "=== BENCHMARK B — scanpy QC CPU reference ==="
echo "Running refs/qc_scanpy_ref.py --scale all ..."

SCANPY_QC_OUT=/dev/shm/c86_scanpy_qc_${SLURM_JOB_ID}.txt
python3 "${REFS_DIR}/qc_scanpy_ref.py" --scale all \
    2>/dev/shm/c86_scanpy_qc_stderr_${SLURM_JOB_ID}.txt \
    | tee "${SCANPY_QC_OUT}"
QC_PY_EXIT=$?
echo "QC_PY_EXIT=${QC_PY_EXIT}"

echo ""
echo "--- scanpy QC stderr ---"
tail -20 /dev/shm/c86_scanpy_qc_stderr_${SLURM_JOB_ID}.txt || true

grep "^SCANPY_QC_WALL_MS=" "${SCANPY_QC_OUT}" || true

# ─── Benchmark B: GPU QC metrics ──────────────────────────────────────────────
echo ""
echo "=== BENCHMARK B — GPU QC metrics (bench_qc_perf) ==="
"${BUILD_DIR}/bench/bench_qc_perf" \
    2>&1 | tee /dev/shm/c86_qc_gpu_${SLURM_JOB_ID}.txt
QC_GPU_EXIT=$?
echo "QC_GPU_EXIT=${QC_GPU_EXIT}"

# ─── Aggregate results ────────────────────────────────────────────────────────
echo ""
echo "=== AGGREGATED RESULTS ==="
echo ""
echo "--- Benchmark A: NMF GPU vs sklearn ---"
echo "GPU timings:"
grep "^BENCH nmf_c86" /dev/shm/c86_nmf_phaseE_${SLURM_JOB_ID}.txt || true
echo ""
echo "sklearn CPU timings:"
grep "^SKLEARN_WALL_MS=" "${SKLEARN_OUT}" || true

# Print combined table for easy registry comparison.
echo ""
echo "=== NMF Phase E Summary Table ==="
echo "scale     | k   | GPU_med_ms | sklearn_ms | ratio"
echo "----------|-----|------------|------------|------"

for scale in small medium; do
    for k in 10 20 50 100; do
        gpu_line=$(grep "scale=${scale}.*k=${k}" /dev/shm/c86_nmf_phaseE_${SLURM_JOB_ID}.txt 2>/dev/null | head -1 || echo "")
        gpu_ms=$(echo "$gpu_line" | grep -oP 'med=\K[\d.]+' || echo "N/A")
        sk_ms=$(grep "SKLEARN_WALL_MS=${scale}_k${k}=" "${SKLEARN_OUT}" 2>/dev/null | cut -d= -f3 || echo "N/A")
        if [ "$gpu_ms" != "N/A" ] && [ "$sk_ms" != "N/A" ] && [ "$sk_ms" != "-1.0" ]; then
            ratio=$(python3 -c "print(f'{float(\"${sk_ms}\")/float(\"${gpu_ms}\"):.2f}x')" 2>/dev/null || echo "N/A")
        else
            ratio="N/A"
        fi
        echo "${scale}     | ${k}   | ${gpu_ms}       | ${sk_ms}      | ${ratio}"
    done
done

echo ""
echo "--- Benchmark B: QC GPU vs scanpy ---"
echo "GPU timings:"
grep "^BENCH qc" /dev/shm/c86_qc_gpu_${SLURM_JOB_ID}.txt || true
echo ""
echo "scanpy timings:"
grep "^SCANPY_QC_WALL_MS=" "${SCANPY_QC_OUT}" || true

echo ""
echo "=== k=20 REGRESSION CHECK ==="
k20_med=$(grep "scale=medium.*k= 20" /dev/shm/c86_nmf_phaseE_${SLURM_JOB_ID}.txt 2>/dev/null | \
          grep -oP 'med=\K[\d.]+' | head -1 || echo "N/A")
echo "k=20 medium GPU median: ${k20_med}ms"
echo "C85 reference: 391ms  Max allowed (10% slack): 430ms"
if [ "$k20_med" != "N/A" ]; then
    python3 -c "
v = float('${k20_med}')
ref = 391.0
cap = ref * 1.10
if v <= cap:
    print(f'REGRESSION: PASS ({v:.1f}ms <= {cap:.0f}ms)')
else:
    print(f'REGRESSION: FAIL ({v:.1f}ms > {cap:.0f}ms, exceeds 10% slack)')
" 2>/dev/null || true
fi

# ─── Device memory snapshot ───────────────────────────────────────────────────
echo ""
echo "--- Device memory after all runs ---"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

# ─── Final summary ────────────────────────────────────────────────────────────
echo ""
echo "=== FINAL EXIT CODES ==="
echo "CMAKE_EXIT=${CMAKE_EXIT}"
echo "BUILD_EXIT=${BUILD_EXIT}"
echo "SKLEARN_EXIT=${SKLEARN_EXIT}"
echo "NMF_EXIT=${NMF_EXIT}"
echo "QC_PY_EXIT=${QC_PY_EXIT}"
echo "QC_GPU_EXIT=${QC_GPU_EXIT}"
date

echo ""
echo "Registry rows appended automatically by bench drivers to:"
echo "  singlet-gpu/state/benchmark-registry.md"
wc -l /mnt/home/debruinz/Singlet-AI/singlet-gpu/state/benchmark-registry.md
