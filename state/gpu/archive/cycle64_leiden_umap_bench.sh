#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/state/cycle64_leiden_umap_bench.sh
#
# Cycle 64 — Features 9 (Leiden) + 10 (UMAP) Phase E:
#   compile + correctness tests + benchmark drivers for both kernels.
#
# Scope:
#   - gpu partition; 1 GPU, 16 CPUs, 64 GB RAM, 2h walltime.
#   - Build: graph_leiden_correctness + embed_umap_correctness (correctness suites).
#   - Build: bench_graph_leiden_perf + bench_embed_umap_perf (bench drivers).
#   - ctest: all leiden / umap correctness tests.
#     GTEST_SKIP expected if cuGraph (Leiden) or cuml (UMAP) headers are absent.
#   - Run bench drivers if builds succeed; capture output.
#   - Append rows to state/benchmark-registry.md.
#   - Write episode to state/cycle-log.md.
#
# Submit: sbatch singlet-gpu/state/cycle64_leiden_umap_bench.sh
#
#SBATCH --job-name=singlet-gpu-cycle64-leiden-umap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle64_leiden_umap_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle64_leiden_umap_%j.log

set -uo pipefail

SINGLET_ROOT="/mnt/home/debruinz/Singlet-AI/singlet-gpu"
BUILD_DIR="${SINGLET_ROOT}/build"
STATE_DIR="${SINGLET_ROOT}/state"
LOG_DIR="${SINGLET_ROOT}/bench/logs"
FACTORNET_ROOT="/mnt/home/debruinz/factornet"
EIGEN3_SPACK="/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3"

# Canonical test sample.
PZ_SAMPLE="/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/gene_counts.1pz"

TODAY=$(date +%Y-%m-%d)

mkdir -p "${LOG_DIR}"

echo "=== Cycle 64: Features 9 (Leiden) + 10 (UMAP) Phase E — compile + correctness + bench ==="
echo "Node: $(hostname)   JobID: ${SLURM_JOB_ID:-local}   Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

# ---------------------------------------------------------------------------
# Step 0: Activate toolchain.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 0: Toolchain activation ---"
source /opt/rh/gcc-toolset-13/enable 2>/dev/null || true
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
gcc --version | head -1
nvcc --version | grep release || echo "nvcc not found"

# ---------------------------------------------------------------------------
# Step 1: Environment / RAPIDS library check.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Environment check (RAPIDS / cuGraph / cuml) ---"
python3 --version
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null        || echo "  numpy: not available"
python3 -c "import scipy; print('scipy', scipy.__version__)" 2>/dev/null        || echo "  scipy: not available"
python3 -c "import cuml; print('cuml: available')" 2>/dev/null                  || echo "  cuml: not available (UMAP bench baseline will be skipped)"
python3 -c "import cugraph; print('cugraph: available')" 2>/dev/null            || echo "  cugraph: not available (Leiden bench baseline will be skipped)"
python3 -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null  || echo "  sklearn: not available"

# Check for cuGraph / cuml C++ headers (determines whether tests skip).
CUGRAPH_HEADER_FOUND=0
CUML_HEADER_FOUND=0
for d in /usr /usr/local /opt/conda /opt/rapids /opt/nvidia; do
    [ -f "${d}/include/cugraph/leiden_clustering.hpp" ] && CUGRAPH_HEADER_FOUND=1
    [ -f "${d}/include/cuml/manifold/umap.hpp" ]        && CUML_HEADER_FOUND=1
done
echo "cuGraph C++ headers found: ${CUGRAPH_HEADER_FOUND}"
echo "cuml C++ headers found:    ${CUML_HEADER_FOUND}"
echo "(GTEST_SKIP expected in tests if headers absent)"

# ---------------------------------------------------------------------------
# Step 2: CMake configure.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: CMake configure ---"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SINGLET_ROOT}" \
    -DFACTORNET_ROOT="${FACTORNET_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
    -DEIGEN_INCLUDE_DIR="${EIGEN3_SPACK}" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=ON \
    2>&1 | tail -20
CMAKE_EXIT=$?
echo "cmake exit: ${CMAKE_EXIT}"
if [ "${CMAKE_EXIT}" -ne 0 ]; then
    echo "ERROR: CMake configure failed — aborting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: Build correctness targets.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3a: Build graph_leiden_correctness ---"
make -j8 graph_leiden_correctness 2>&1 | tail -20
LEIDEN_BUILD=$?
echo "graph_leiden_correctness build exit: ${LEIDEN_BUILD}"
[ "${LEIDEN_BUILD}" -eq 0 ] && echo "graph_leiden_correctness: BUILD OK" || echo "COMPILE FAILED: graph_leiden_correctness"

echo ""
echo "--- Step 3b: Build embed_umap_correctness ---"
make -j8 embed_umap_correctness 2>&1 | tail -20
UMAP_BUILD=$?
echo "embed_umap_correctness build exit: ${UMAP_BUILD}"
[ "${UMAP_BUILD}" -eq 0 ] && echo "embed_umap_correctness: BUILD OK" || echo "COMPILE FAILED: embed_umap_correctness"

# ---------------------------------------------------------------------------
# Step 4: Build bench drivers.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4a: Build bench_graph_leiden_perf ---"
make -j8 bench_graph_leiden_perf 2>&1 | tail -10
LEIDEN_BENCH_BUILD=$?
echo "bench_graph_leiden_perf build exit: ${LEIDEN_BENCH_BUILD}"
[ "${LEIDEN_BENCH_BUILD}" -eq 0 ] && LEIDEN_BENCH_AVAILABLE=1 || LEIDEN_BENCH_AVAILABLE=0

echo ""
echo "--- Step 4b: Build bench_embed_umap_perf ---"
make -j8 bench_embed_umap_perf 2>&1 | tail -10
UMAP_BENCH_BUILD=$?
echo "bench_embed_umap_perf build exit: ${UMAP_BENCH_BUILD}"
[ "${UMAP_BENCH_BUILD}" -eq 0 ] && UMAP_BENCH_AVAILABLE=1 || UMAP_BENCH_AVAILABLE=0

cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 5: ctest — Leiden correctness.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: ctest graph_leiden ---"
cd "${BUILD_DIR}"

LEIDEN_CTEST_LOG="/tmp/cycle64_ctest_leiden.log"
if [ "${LEIDEN_BUILD}" -eq 0 ]; then
    ctest -R "leiden|Leiden" -V --timeout 300 2>&1 | tee "${LEIDEN_CTEST_LOG}"
    LEIDEN_CTEST_EXIT=$?
else
    echo "Skipping Leiden ctest (build failed)."
    LEIDEN_CTEST_EXIT=1
    touch "${LEIDEN_CTEST_LOG}"
fi

LEIDEN_PASS=$(grep -c "PASSED\|Passed" "${LEIDEN_CTEST_LOG}" 2>/dev/null || true)
LEIDEN_TOTAL=$(grep -c "Test #" "${LEIDEN_CTEST_LOG}" 2>/dev/null || true)
LEIDEN_SKIP=$(grep -c "GTEST_SKIP\|SKIPPED\|cuGraph not found\|cugraph" "${LEIDEN_CTEST_LOG}" 2>/dev/null || true)

echo ""
echo "ctest leiden exit: ${LEIDEN_CTEST_EXIT}"
echo "ctest leiden: ${LEIDEN_PASS} passed / ${LEIDEN_TOTAL} total  (GTEST_SKIP count: ${LEIDEN_SKIP})"

cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 6: ctest — UMAP correctness.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: ctest embed_umap ---"
cd "${BUILD_DIR}"

UMAP_CTEST_LOG="/tmp/cycle64_ctest_umap.log"
if [ "${UMAP_BUILD}" -eq 0 ]; then
    ctest -R "umap|Umap|UMAP" -V --timeout 300 2>&1 | tee "${UMAP_CTEST_LOG}"
    UMAP_CTEST_EXIT=$?
else
    echo "Skipping UMAP ctest (build failed)."
    UMAP_CTEST_EXIT=1
    touch "${UMAP_CTEST_LOG}"
fi

UMAP_PASS=$(grep -c "PASSED\|Passed" "${UMAP_CTEST_LOG}" 2>/dev/null || true)
UMAP_TOTAL=$(grep -c "Test #" "${UMAP_CTEST_LOG}" 2>/dev/null || true)
UMAP_SKIP=$(grep -c "GTEST_SKIP\|SKIPPED\|cuml not found\|cuml" "${UMAP_CTEST_LOG}" 2>/dev/null || true)

echo ""
echo "ctest umap exit: ${UMAP_CTEST_EXIT}"
echo "ctest umap: ${UMAP_PASS} passed / ${UMAP_TOTAL} total  (GTEST_SKIP count: ${UMAP_SKIP})"

cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 7: Run bench_graph_leiden_perf (if available).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 7: bench_graph_leiden_perf ---"

LEIDEN_BENCH_BIN="${BUILD_DIR}/bench/bench_graph_leiden_perf"
LEIDEN_BENCH_EXIT=1

if [ "${LEIDEN_BENCH_AVAILABLE}" -eq 1 ] && [ -f "${LEIDEN_BENCH_BIN}" ]; then
    echo "Sample path: ${PZ_SAMPLE}"
    [ -f "${PZ_SAMPLE}" ] || echo "WARNING: canonical sample not found — bench driver may use synthetic fallback."
    echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${LEIDEN_BENCH_BIN}" 2>&1
    LEIDEN_BENCH_EXIT=$?
    echo "bench_graph_leiden_perf exit: ${LEIDEN_BENCH_EXIT} — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
    echo "bench_graph_leiden_perf binary not available — skipping."
fi

# ---------------------------------------------------------------------------
# Step 8: Run bench_embed_umap_perf (if available).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 8: bench_embed_umap_perf ---"

UMAP_BENCH_BIN="${BUILD_DIR}/bench/bench_embed_umap_perf"
UMAP_BENCH_EXIT=1

if [ "${UMAP_BENCH_AVAILABLE}" -eq 1 ] && [ -f "${UMAP_BENCH_BIN}" ]; then
    echo "Sample path: ${PZ_SAMPLE}"
    [ -f "${PZ_SAMPLE}" ] || echo "WARNING: canonical sample not found — bench driver may use synthetic fallback."
    echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${UMAP_BENCH_BIN}" 2>&1
    UMAP_BENCH_EXIT=$?
    echo "bench_embed_umap_perf exit: ${UMAP_BENCH_EXIT} — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
    echo "bench_embed_umap_perf binary not available — skipping."
fi

# ---------------------------------------------------------------------------
# Step 9: benchmark-registry.md append.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 9: benchmark-registry.md append ---"

python3 - <<PYEOF
import os
from datetime import date

today = date.today().isoformat()

leiden_log = "/tmp/cycle64_ctest_leiden.log"
umap_log   = "/tmp/cycle64_ctest_umap.log"

def parse_ctest(log_path):
    n_pass = n_total = n_skip = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "Test #" in line:
                    n_total += 1
                if "PASSED" in line or "Passed" in line:
                    n_pass += 1
                if "GTEST_SKIP" in line or "SKIPPED" in line:
                    n_skip += 1
    return n_pass, n_total, n_skip

lp, lt, ls = parse_ctest(leiden_log)
up, ut, us = parse_ctest(umap_log)

leiden_str = f"{lp}/{lt} pass ({ls} cuGraph GTEST_SKIP)"
umap_str   = f"{up}/{ut} pass ({us} cuml GTEST_SKIP)"

leiden_build = "${LEIDEN_BUILD}"
umap_build   = "${UMAP_BUILD}"
leiden_bench = "${LEIDEN_BENCH_BUILD}"
umap_bench   = "${UMAP_BENCH_BUILD}"
leiden_bench_run = "${LEIDEN_BENCH_EXIT}"
umap_bench_run   = "${UMAP_BENCH_EXIT}"

rows = [
    f"| {today} | graph/leiden ctest | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: {leiden_str}; build={leiden_build}; bench_build={leiden_bench}; bench_run={leiden_bench_run} | cycle64 |",
    f"| {today} | embed/umap ctest   | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — | ctest: {umap_str};   build={umap_build};   bench_build={umap_bench};   bench_run={umap_bench_run}   | cycle64 |",
]

registry_path = "${STATE_DIR}/benchmark-registry.md"
try:
    with open(registry_path, "a") as f:
        for row in rows:
            f.write(row + "\n")
    print(f"  benchmark-registry.md: {len(rows)} rows appended.")
except Exception as e:
    print(f"  WARNING: could not write registry: {e}")

print(f"  Leiden ctest: {leiden_str}")
print(f"  UMAP   ctest: {umap_str}")
PYEOF

# ---------------------------------------------------------------------------
# Step 10: cycle-log.md episode entry.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 10: cycle-log.md episode ---"

python3 - <<PYEOF
import os
from datetime import datetime, timezone

now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

leiden_log = "/tmp/cycle64_ctest_leiden.log"
umap_log   = "/tmp/cycle64_ctest_umap.log"

def parse_ctest(log_path):
    n_pass = n_total = n_skip = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "Test #" in line:
                    n_total += 1
                if "PASSED" in line or "Passed" in line:
                    n_pass += 1
                if "GTEST_SKIP" in line or "SKIPPED" in line:
                    n_skip += 1
    return n_pass, n_total, n_skip

lp, lt, ls = parse_ctest(leiden_log)
up, ut, us = parse_ctest(umap_log)

leiden_build      = "${LEIDEN_BUILD}"
umap_build        = "${UMAP_BUILD}"
leiden_bench_bld  = "${LEIDEN_BENCH_BUILD}"
umap_bench_bld    = "${UMAP_BENCH_BUILD}"
leiden_ctest_exit = "${LEIDEN_CTEST_EXIT}"
umap_ctest_exit   = "${UMAP_CTEST_EXIT}"
leiden_bench_run  = "${LEIDEN_BENCH_EXIT}"
umap_bench_run    = "${UMAP_BENCH_EXIT}"

lines = [
    f"\n## Cycle 64 ({now_str}) — Features 9 (Leiden) + 10 (UMAP) Phase E: compile + correctness + bench\n",
    f"- **Feature 9 (graph/leiden.h)**: wraps cuGraph Leiden clustering\n",
    f"  - build exit: {leiden_build}\n",
    f"  - ctest exit: {leiden_ctest_exit}  |  {lp}/{lt} passed, {ls} cuGraph GTEST_SKIP\n",
    f"  - bench_graph_leiden_perf build: {leiden_bench_bld}  |  run exit: {leiden_bench_run}\n",
    f"- **Feature 10 (embed/umap.h)**: wraps cuml UMAP\n",
    f"  - build exit: {umap_build}\n",
    f"  - ctest exit: {umap_ctest_exit}  |  {up}/{ut} passed, {us} cuml GTEST_SKIP\n",
    f"  - bench_embed_umap_perf build: {umap_bench_bld}  |  run exit: {umap_bench_run}\n",
    f"- **Node**: $(hostname)   JobID: ${SLURM_JOB_ID:-local}\n",
    f"- **Next cycle**: proceed to Feature 11 (DE) or next DAG task per dag.md\n",
]

cycle_log = "${STATE_DIR}/cycle-log.md"
with open(cycle_log, "a") as f:
    f.writelines(lines)

print(f"cycle-log.md: Cycle 64 episode written.")
PYEOF

# ---------------------------------------------------------------------------
# Step 11: Final summary.
# ---------------------------------------------------------------------------
echo ""
echo "=== Cycle 64 final summary ==="
echo "Date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Node:  $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-local}"
echo ""
echo "LEIDEN:"
echo "  correctness build: exit ${LEIDEN_BUILD}"
echo "  bench build:       exit ${LEIDEN_BENCH_BUILD}"
echo "  ctest exit:        ${LEIDEN_CTEST_EXIT}"
echo "  ctest result:      ${LEIDEN_PASS:-0}/${LEIDEN_TOTAL:-0} passed  (${LEIDEN_SKIP:-0} cuGraph GTEST_SKIPs)"
echo "  bench driver:      exit ${LEIDEN_BENCH_EXIT}"
echo ""
echo "UMAP:"
echo "  correctness build: exit ${UMAP_BUILD}"
echo "  bench build:       exit ${UMAP_BENCH_BUILD}"
echo "  ctest exit:        ${UMAP_CTEST_EXIT}"
echo "  ctest result:      ${UMAP_PASS:-0}/${UMAP_TOTAL:-0} passed  (${UMAP_SKIP:-0} cuml GTEST_SKIPs)"
echo "  bench driver:      exit ${UMAP_BENCH_EXIT}"
echo ""
echo "--- benchmark-registry.md (last 10 rows) ---"
tail -10 "${STATE_DIR}/benchmark-registry.md" || true
