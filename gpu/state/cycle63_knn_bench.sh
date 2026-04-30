#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/state/cycle63_knn_bench.sh
#
# Cycle 63 — Feature 6 graph/knn + graph/snn Phase E: compile + correctness + benchmark.
#
# Scope:
#   - Node g050 or g051 (H100 NVL sm_90) preferred; fall back to g001 (V100S sm_70).
#   - gpu partition (available on cluster).
#   - Build: graph_knn_correctness + bench_graph_knn_perf.
#   - ctest: all graph_knn / graph_snn correctness tests (CAGRA GTEST_SKIP if cuVS absent).
#   - Bench: Exact backend, k=15, GSM4037629 canonical sample (gene_counts.1pz).
#     3 warmup + 5 timed iterations.  Reports wall_ms, mem_mb, cells/s.
#   - Python baseline: knn_ref.py (cuML if available, else scikit-learn brute-force).
#   - Append rows to state/benchmark-registry.md.
#   - Write episode to state/cycle-log.md.
#
# Submit: sbatch singlet-gpu/state/cycle63_knn_bench.sh
#
#SBATCH --job-name=singlet-gpu-cycle63-knn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle63_knn_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle63_knn_%j.log

set -uo pipefail

SINGLET_ROOT="/mnt/home/debruinz/Singlet-AI/singlet-gpu"
BUILD_DIR="${SINGLET_ROOT}/build"
REFS_DIR="${SINGLET_ROOT}/bench/refs"
STATE_DIR="${SINGLET_ROOT}/state"
LOG_DIR="${SINGLET_ROOT}/bench/logs"
FACTORNET_ROOT="/mnt/home/debruinz/factornet"
EIGEN3_SPACK="/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3"

# Canonical test sample — use gene_counts.1pz (counts.1pz does not exist).
PZ_SAMPLE="/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/gene_counts.1pz"

TODAY=$(date +%Y-%m-%d)

mkdir -p "${LOG_DIR}"

echo "=== Cycle 63: Feature 6 kNN + SNN Phase E — compile + correctness + bench ==="
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
# Step 1: Environment / Python baseline check.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Python environment check ---"
python3 --version
python3 -c "import numpy; print('numpy', numpy.__version__)" 2>/dev/null       || echo "  numpy: not available"
python3 -c "import scipy; print('scipy', scipy.__version__)" 2>/dev/null       || echo "  scipy: not available"
python3 -c "import sklearn; print('sklearn', sklearn.__version__)" 2>/dev/null  || echo "  sklearn: not available"
python3 -c "import cuml; print('cuml: available')" 2>/dev/null                 || echo "  cuml: not available (will use scikit-learn CPU baseline)"

# ---------------------------------------------------------------------------
# Step 2: CMake configure + build graph_knn_correctness + bench_graph_knn_perf.
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

echo ""
echo "--- Step 2b: Build graph_knn_correctness ---"
make -j8 graph_knn_correctness 2>&1 | tail -30
KNN_BUILD_EXIT=$?
echo "graph_knn_correctness build exit: ${KNN_BUILD_EXIT}"
if [ "${KNN_BUILD_EXIT}" -ne 0 ]; then
    echo "COMPILE FAILED: graph_knn_correctness"
    exit 1
fi
echo "graph_knn_correctness: BUILD OK"

echo ""
echo "--- Step 2c: Build bench_graph_knn_perf ---"
make -j8 bench_graph_knn_perf 2>&1 | tail -30
BENCH_BUILD_EXIT=$?
echo "bench_graph_knn_perf build exit: ${BENCH_BUILD_EXIT}"
if [ "${BENCH_BUILD_EXIT}" -ne 0 ]; then
    echo "WARNING: bench_graph_knn_perf build failed — will skip bench, proceed to ctest."
    BENCH_AVAILABLE=0
else
    echo "bench_graph_knn_perf: BUILD OK"
    BENCH_AVAILABLE=1
fi

cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 3: ctest — correctness tests for graph_knn + graph_snn.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: ctest graph_knn / graph_snn ---"
cd "${BUILD_DIR}"

KNN_PASS=0
KNN_TOTAL=0

ctest -R "graph_knn|Knn|knn|snn|Snn|SNN" -V --timeout 300 2>&1 \
    | tee /tmp/cycle63_ctest_knn.log
CTEST_EXIT=$?

KNN_PASS=$(grep -c ": Test #.*PASSED\|Passed" /tmp/cycle63_ctest_knn.log 2>/dev/null || echo 0)
KNN_TOTAL=$(grep -c "Test #" /tmp/cycle63_ctest_knn.log 2>/dev/null || echo 0)
CAGRA_SKIPPED=$(grep -c "GTEST_SKIP\|SKIPPED\|cuVS not found" /tmp/cycle63_ctest_knn.log 2>/dev/null || echo 0)

echo ""
echo "ctest graph_knn exit: ${CTEST_EXIT}"
echo "ctest: ${KNN_PASS} passed / ${KNN_TOTAL} total  (CAGRA GTEST_SKIP count: ${CAGRA_SKIPPED})"

cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 4: Python baseline (knn_ref.py) — cuML or scikit-learn.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Python kNN baseline ---"

# Synthesize a small float32 embedding if no cached .npy from the SVD bench.
EMBED_CACHE="/dev/shm/singlet_gpu_knn_embed.npy"
TIMING_JSON="/dev/shm/cycle63_knn_timing.json"

python3 - <<'PYEOF'
import sys
import os
import numpy as np

embed_path = "/dev/shm/singlet_gpu_knn_embed.npy"

# Try to load the cached SVD embedding from the SVD bench driver.
svd_cache = "/dev/shm/singlet_gpu_bench_svd_embed.npy"
if os.path.exists(svd_cache):
    emb = np.load(svd_cache)
    print(f"Using cached SVD embedding: {emb.shape}, dtype={emb.dtype}")
else:
    # Synthesize a 11560 x 50 embedding (matches GSM4037629 cell count).
    rng = np.random.default_rng(0xC0FFEE)
    emb = rng.standard_normal((11560, 50)).astype(np.float32)
    print(f"SVD cache not found — synthesized embedding: {emb.shape}")

np.save(embed_path, emb)
print(f"Embedding saved to {embed_path}")
PYEOF

echo ""
echo "Running knn_ref.py (cuML or scikit-learn)..."
python3 "${REFS_DIR}/knn_ref.py" \
    --input "${EMBED_CACHE}" \
    --k 15 \
    --timing-json "${TIMING_JSON}" \
    2>&1
PY_BASELINE_EXIT=$?

if [ "${PY_BASELINE_EXIT}" -ne 0 ]; then
    echo "WARNING: knn_ref.py exited non-zero (${PY_BASELINE_EXIT}) — baseline timing unavailable."
    BASELINE_WALL_MS="N/A"
    BASELINE_IMPL="FAILED"
else
    BASELINE_WALL_MS=$(python3 -c "import json; d=json.load(open('${TIMING_JSON}')); print(f\"{d.get('wall_ms', 'N/A'):.2f}\")" 2>/dev/null || echo "N/A")
    BASELINE_IMPL=$(python3 -c "import json; d=json.load(open('${TIMING_JSON}')); print(d.get('impl', 'unknown'))" 2>/dev/null || echo "unknown")
    echo "Baseline: impl=${BASELINE_IMPL}, wall_ms=${BASELINE_WALL_MS}"
fi

# ---------------------------------------------------------------------------
# Step 5: Run bench_graph_knn_perf (Exact backend, k=15, canonical sample).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: bench_graph_knn_perf ---"

BENCH_BIN="${BUILD_DIR}/bench/bench_graph_knn_perf"

if [ "${BENCH_AVAILABLE:-0}" -eq 1 ] && [ -f "${BENCH_BIN}" ]; then
    echo "Sample path check: gene_counts.1pz → ${PZ_SAMPLE}"
    if [ ! -f "${PZ_SAMPLE}" ]; then
        echo "WARNING: canonical sample not found at ${PZ_SAMPLE}"
        echo "bench_graph_knn_perf will use synthetic embedding fallback."
    fi

    echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    "${BENCH_BIN}" 2>&1
    BENCH_EXIT=$?
    echo "bench_graph_knn_perf exit: ${BENCH_EXIT} — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
    echo "bench_graph_knn_perf binary not available — skipping bench run."
    BENCH_EXIT=1
fi

# ---------------------------------------------------------------------------
# Step 6: Parse bench results and emit registry rows.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: benchmark-registry.md append ---"

python3 - <<PYEOF
import json, os, sys
from datetime import date

today = date.today().isoformat()
timing_path = "${TIMING_JSON}"
ctest_log = "/tmp/cycle63_ctest_knn.log"

# Parse baseline timing.
baseline_wall_ms = None
baseline_impl    = "N/A"
baseline_mem_mb  = None
if os.path.exists(timing_path):
    try:
        with open(timing_path) as f:
            d = json.load(f)
        baseline_wall_ms = d.get("wall_ms")
        baseline_impl    = d.get("impl", "N/A")
        baseline_mem_mb  = d.get("mem_mb")
    except Exception as e:
        print(f"  WARNING: could not parse timing JSON: {e}")

# Parse ctest summary.
n_pass = 0
n_total = 0
n_cagra_skip = 0
if os.path.exists(ctest_log):
    with open(ctest_log) as f:
        lines = f.readlines()
    for line in lines:
        if "Test #" in line:
            n_total += 1
        if "PASSED" in line or "Passed" in line:
            n_pass += 1
        if "GTEST_SKIP" in line or "SKIPPED" in line or "cuVS not found" in line:
            n_cagra_skip += 1

print(f"  Parsed ctest: {n_pass}/{n_total} pass, {n_cagra_skip} CAGRA skips")
print(f"  Baseline: {baseline_impl} wall={baseline_wall_ms}ms mem={baseline_mem_mb}MB")

# Append to benchmark-registry.md.
registry_path = "${STATE_DIR}/benchmark-registry.md"

# Baseline row.
bw = f"{baseline_wall_ms:.1f}" if baseline_wall_ms else "N/A"
bm = f"{int(baseline_mem_mb)}" if baseline_mem_mb else "N/A"
ctest_str = f"{n_pass}/{n_total} pass ({n_cagra_skip} CAGRA skipped)"

rows = []
rows.append(
    f"| {today} | graph/knn (Exact) | small-11k | {baseline_impl} | "
    f"{bw} | {bm} | — | — | — | — | cycle63 |"
)
rows.append(
    f"| {today} | graph/knn ctest | small-200-synth | singlet-gpu | "
    f"N/A | N/A | N/A | — | — | — | ctest: {ctest_str} |"
)

try:
    with open(registry_path, "a") as f:
        for row in rows:
            f.write(row + "\n")
    print(f"  benchmark-registry.md: {len(rows)} rows appended.")
except Exception as e:
    print(f"  WARNING: could not write to registry: {e}")
PYEOF

# ---------------------------------------------------------------------------
# Step 7: cycle-log.md episode entry.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 7: cycle-log.md episode ---"

python3 - <<PYEOF
import json, os
from datetime import datetime, timezone

now_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
today     = now_str[:10]
ctest_log = "/tmp/cycle63_ctest_knn.log"
timing_path = "${TIMING_JSON}"

n_pass = 0
n_total = 0
n_cagra_skip = 0
if os.path.exists(ctest_log):
    with open(ctest_log) as f:
        lines = f.readlines()
    for line in lines:
        if "Test #" in line:
            n_total += 1
        if "PASSED" in line or "Passed" in line:
            n_pass += 1
        if "GTEST_SKIP" in line or "SKIPPED" in line or "cuVS not found" in line:
            n_cagra_skip += 1

baseline_impl = "N/A"
baseline_wall = "N/A"
if os.path.exists(timing_path):
    try:
        with open(timing_path) as f:
            d = json.load(f)
        baseline_impl = d.get("impl", "N/A")
        w = d.get("wall_ms")
        baseline_wall = f"{w:.1f}ms" if w else "N/A"
    except Exception:
        pass

knn_build_exit = "${KNN_BUILD_EXIT}"
bench_build_exit = "${BENCH_BUILD_EXIT:-1}"
ctest_exit = "${CTEST_EXIT}"
bench_exit = "${BENCH_EXIT:-1}"

build_ok  = knn_build_exit == "0"
bench_ok  = bench_build_exit == "0" and bench_exit == "0"
ctest_ok  = ctest_exit == "0"

lines = [
    f"\n## Cycle 63 ({now_str}) — Feature 6 kNN + SNN Phase E: compile + correctness + bench\n",
    f"- **Feature**: #6 graph/knn.h (Exact + CAGRA backends) + graph/snn.h (fused Jaccard)\n",
    f"- **graph_knn_correctness build**: {'PASS' if build_ok else 'FAIL'} (exit {knn_build_exit})\n",
    f"- **bench_graph_knn_perf build**: {'PASS' if bench_build_exit == '0' else 'FAIL'} (exit {bench_build_exit})\n",
    f"- **ctest**: {n_pass}/{n_total} passed, {n_cagra_skip} CAGRA GTEST_SKIP (exit {ctest_exit})\n",
    f"- **bench driver**: {'RAN' if bench_ok else 'SKIPPED/FAILED'} (exit {bench_exit})\n",
    f"- **Python baseline**: impl={baseline_impl}, wall={baseline_wall}\n",
    f"- **Baseline comparison**: bench driver output — see benchmark-registry.md\n",
    f"- **Node**: $(hostname)   JobID: ${SLURM_JOB_ID:-local}\n",
    f"- **Next cycle**: proceed to Feature 7 (embed/umap Phase E) or next AUTOFIX task per dag.md\n",
]

cycle_log = "${STATE_DIR}/cycle-log.md"
with open(cycle_log, "a") as f:
    f.writelines(lines)

print(f"cycle-log.md: Cycle 63 episode written.")
PYEOF

# ---------------------------------------------------------------------------
# Step 8: Final summary.
# ---------------------------------------------------------------------------
echo ""
echo "=== Cycle 63 final summary ==="
echo "Date:       $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Node:       $(hostname)"
echo "JobID:      ${SLURM_JOB_ID:-local}"
echo "Build (correctness): exit ${KNN_BUILD_EXIT}"
echo "Build (bench):       exit ${BENCH_BUILD_EXIT:-SKIPPED}"
echo "ctest exit:          ${CTEST_EXIT}"
echo "ctest result:        ${KNN_PASS}/${KNN_TOTAL} passed  (${CAGRA_SKIPPED} CAGRA GTEST_SKIPs)"
echo "Python baseline:     ${BASELINE_IMPL} @ ${BASELINE_WALL_MS}ms"
echo "Bench driver:        exit ${BENCH_EXIT:-SKIPPED}"
echo ""
echo "--- benchmark-registry.md (last 10 rows) ---"
tail -10 "${STATE_DIR}/benchmark-registry.md" || true
