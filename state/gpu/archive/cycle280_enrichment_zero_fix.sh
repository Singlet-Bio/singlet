#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 280 — build + verify enrichment/GSEA zero-output bug fixes
#   CYCLE-122-ENRICHMENT-ZERO-OUTPUT-DIAG
#
# Targets:
#   gsea_aucell_correctness          → AUCell_RealData_RanksConsistent
#   gsea_fgsea_correctness           → Fgsea_BHCorrection_NotAllZero
#   enrich_ssgsea_progeny_correctness → Ssgsea_GSM_RealData, Progeny_HumanTop100_RealData
#
# Fixes applied in this cycle:
#   aucell.h    — aucell_score_kernel warp-parallel scan had prefix/suffix direction
#                 reversed; replaced with serial lane-0 scan.
#   fgsea.h     — abs_kernel / scatter_lut_kernel / set_ones_kernel were declared
#                 but never launched; BH SortPairs used in-place alias for d_idx.
#   enrich_ssgsea_progeny_correctness.cpp — progeny_run stub now implements GEMM
#                 + Welford normalization; Ssgsea_GSM_RealData calls real ssgsea() API.

#SBATCH --job-name=sg_c280_enrich
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=g001,g002,g005
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle280_enrichment_zero_fix_%j.log

set -uo pipefail

# ---- Environment ---------------------------------------------------------------
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc
CUDA_ARCH="70;80;90"

BUILD_DIR=/mnt/projects/debruinz_project/singlet-gpu/build/cycle88_verify2
SRC_DIR=/mnt/home/debruinz/Singlet-AI/singlet-gpu
EIGEN_DIR=/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3

echo "=== Cycle 280: enrichment zero-output bug fixes ==="
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)"
echo "Date: $(date)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "g++:  $(${CXX} --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"

# ---- Fresh build dir -----------------------------------------------------------
rm -rf "$BUILD_DIR"

echo ""
echo "--- cmake configure ---"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=OFF \
    -DEIGEN_INCLUDE_DIR="${EIGEN_DIR}" \
    2>&1 | tail -30
CMAKE_EXIT=${PIPESTATUS[0]}
echo "CMAKE_EXIT=${CMAKE_EXIT}"
[ "${CMAKE_EXIT}" -ne 0 ] && { echo "=== CMAKE CONFIGURE FAILED ==="; exit "${CMAKE_EXIT}"; }

# ---- Build the three test targets ----------------------------------------------
echo ""
echo "--- cmake build: gsea_fgsea_correctness gsea_aucell_correctness enrich_ssgsea_progeny_correctness ---"
cmake --build "$BUILD_DIR" \
    --target gsea_fgsea_correctness gsea_aucell_correctness enrich_ssgsea_progeny_correctness \
    -j8 2>&1 | tail -80
BUILD_EXIT=${PIPESTATUS[0]}
echo "BUILD_EXIT=${BUILD_EXIT}"
[ "${BUILD_EXIT}" -ne 0 ] && { echo "=== BUILD FAILED ==="; exit "${BUILD_EXIT}"; }

AUCELL_BIN="$BUILD_DIR/tests/gsea_aucell_correctness"
FGSEA_BIN="$BUILD_DIR/tests/gsea_fgsea_correctness"
SSGSEA_BIN="$BUILD_DIR/tests/enrich_ssgsea_progeny_correctness"

# ---- Run targeted tests --------------------------------------------------------

AUCELL_EXIT=99
FGSEA_EXIT=99
SSGSEA_EXIT=99

# Test 1: AUCell ranks-consistent (mean_high > mean_bg)
if [ -x "$AUCELL_BIN" ]; then
    echo ""
    echo "--- AUCell: RealData_RanksConsistent ---"
    "$AUCELL_BIN" \
        --gtest_filter="AUCell.RealData_RanksConsistent" \
        --gtest_color=no 2>&1 | tail -40
    AUCELL_EXIT=${PIPESTATUS[0]}
fi
echo "AUCELL_EXIT=${AUCELL_EXIT}"

# Test 2: fgsea BH correction (q-values not all zero)
if [ -x "$FGSEA_BIN" ]; then
    echo ""
    echo "--- fgsea: BHCorrection_NotAllZero ---"
    "$FGSEA_BIN" \
        --gtest_filter="Fgsea.BHCorrection_NotAllZero" \
        --gtest_color=no 2>&1 | tail -40
    FGSEA_EXIT=${PIPESTATUS[0]}
fi
echo "FGSEA_EXIT=${FGSEA_EXIT}"

# Test 3: ssGSEA + PROGENy real-data variance (requires GSM file on disk)
if [ -x "$SSGSEA_BIN" ]; then
    echo ""
    echo "--- ssGSEA/PROGENy: real-data variance tests ---"
    "$SSGSEA_BIN" \
        --gtest_filter="SsGseaCorrectness.Ssgsea_GSM_RealData:ProgenyCorrectness.Progeny_HumanTop100_RealData" \
        --gtest_color=no 2>&1 | tail -60
    SSGSEA_EXIT=${PIPESTATUS[0]}
fi
echo "SSGSEA_EXIT=${SSGSEA_EXIT}"

# ---- Summary -------------------------------------------------------------------
echo ""
echo "=== FINAL EXIT CODES ==="
echo "CMAKE_EXIT=${CMAKE_EXIT}"
echo "BUILD_EXIT=${BUILD_EXIT}"
echo "AUCELL_EXIT=${AUCELL_EXIT}"
echo "FGSEA_EXIT=${FGSEA_EXIT}"
echo "SSGSEA_EXIT=${SSGSEA_EXIT}"
echo "Date: $(date)"

# Exit non-zero if any test failed (99 = binary not built; treat as failure too)
OVERALL=0
[ "${AUCELL_EXIT}" -ne 0 ] && OVERALL=1
[ "${FGSEA_EXIT}"  -ne 0 ] && OVERALL=1
[ "${SSGSEA_EXIT}" -ne 0 ] && OVERALL=1
exit "${OVERALL}"
