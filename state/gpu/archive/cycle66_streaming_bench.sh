#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/state/cycle66_streaming_bench.sh
#
# Cycle 66 — Feature 17 Streaming Phase E: GPU compile + correctness tests.
#
# Scope:
#   - gpu partition; 1 GPU, 16 CPUs, 64 GB RAM, 1h walltime.
#   - Build: streaming_pipeline_correctness (+ bench_streaming_perf if it exists).
#   - ctest: all streaming / Streaming / Pipeline correctness tests.
#   - Capture ALL compile errors verbatim — these feed the DAG fix queue.
#   - Known disabled tests (#if 0): Pipeline_NmfChunked, Pipeline_GeneMismatch
#     (blocked on CYCLE-7-FOLLOWUP-WRITE-PZ: write_pz not yet implemented).
#   - Test data: exon_counts.1pz for GSM4037629 (20866 cells, full artifact suite).
#   - Append rows to state/benchmark-registry.md.
#   - Append episode to state/cycle-log.md.
#
# Submit: sbatch singlet-gpu/state/cycle66_streaming_bench.sh
#
#SBATCH --job-name=singlet-gpu-cycle66-streaming
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle66_streaming_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle66_streaming_%j.log

set -uo pipefail

SINGLET_ROOT="/mnt/home/debruinz/Singlet-AI/singlet-gpu"
BUILD_DIR="${SINGLET_ROOT}/build"
STATE_DIR="${SINGLET_ROOT}/state"
LOG_DIR="${SINGLET_ROOT}/bench/logs"
FACTORNET_ROOT="/mnt/home/debruinz/factornet"
EIGEN3_SPACK="/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3"

TEST_PZ="/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz"

TODAY=$(date +%Y-%m-%d)

mkdir -p "${LOG_DIR}"

echo "=== Cycle 66: Feature 17 Streaming Phase E — GPU compile + correctness ==="
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

# Confirm test data exists.
echo ""
if [ -f "${TEST_PZ}" ]; then
    echo "Test data: FOUND — ${TEST_PZ}"
else
    echo "WARNING: Test data NOT found at ${TEST_PZ}"
    echo "Tests will GTEST_SKIP (canonical sample check in SetUp)."
fi

# ---------------------------------------------------------------------------
# Step 1: CMake configure.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: CMake configure ---"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

export FACTORNET_ROOT="${FACTORNET_ROOT}"

cmake "${SINGLET_ROOT}" \
    -DFACTORNET_ROOT="${FACTORNET_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;80;90" \
    -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
    -DEIGEN_INCLUDE_DIR="${EIGEN3_SPACK}" \
    -DSINGLET_GPU_BUILD_TESTS=ON \
    -DSINGLET_GPU_BUILD_BENCH=ON \
    2>&1 | tail -5
CMAKE_EXIT=$?
echo "cmake exit: ${CMAKE_EXIT}"
if [ "${CMAKE_EXIT}" -ne 0 ]; then
    echo "ERROR: CMake configure failed — aborting."
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: Build streaming_pipeline_correctness.
# Capture ALL output (not just tail) so compile errors are fully visible.
# ---------------------------------------------------------------------------
echo ""
echo "=== Building streaming test ==="
BUILD_LOG="/tmp/cycle66_build_streaming.log"

make -j8 streaming_pipeline_correctness 2>&1 | tee "${BUILD_LOG}"
BUILD_EXIT=$?
echo "Build exit: ${BUILD_EXIT}"

if [ "${BUILD_EXIT}" -eq 0 ]; then
    echo "streaming_pipeline_correctness: BUILD OK"
else
    echo "COMPILE FAILED: streaming_pipeline_correctness"
    echo "--- First 80 error lines ---"
    grep -i "error:" "${BUILD_LOG}" | head -80 || true
fi

# ---------------------------------------------------------------------------
# Step 3: Streaming correctness tests via ctest.
# ---------------------------------------------------------------------------
echo ""
echo "=== Streaming ctest ==="
STREAM_CTEST_LOG="/tmp/cycle66_ctest_streaming.log"

if [ "${BUILD_EXIT}" -eq 0 ]; then
    ctest -R "streaming|Streaming|Pipeline" -V --timeout 600 2>&1 | tee "${STREAM_CTEST_LOG}"
    STREAM_CTEST=$?
else
    echo "Skipping streaming ctest (build failed)."
    STREAM_CTEST=1
    touch "${STREAM_CTEST_LOG}"
fi

STREAM_PASS=$(grep -c "PASSED\|Passed" "${STREAM_CTEST_LOG}" 2>/dev/null || true)
STREAM_TOTAL=$(grep -c "Test #" "${STREAM_CTEST_LOG}" 2>/dev/null || true)
STREAM_SKIP=$(grep -c "GTEST_SKIP\|SKIPPED" "${STREAM_CTEST_LOG}" 2>/dev/null || true)
STREAM_WALL=$(grep "Total Test time" "${STREAM_CTEST_LOG}" 2>/dev/null | grep -oP '[\d.]+\s+sec' | tail -1 || echo "n/a")

echo ""
echo "Streaming ctest exit: ${STREAM_CTEST}"
echo "Streaming ctest: ${STREAM_PASS} passed / ${STREAM_TOTAL} total  (GTEST_SKIP count: ${STREAM_SKIP})"
echo "Wall time: ${STREAM_WALL}"

cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 4: Build bench_streaming_perf if target exists.
# ---------------------------------------------------------------------------
echo ""
echo "=== Building streaming bench driver (if target exists) ==="
cd "${BUILD_DIR}"
make -j8 bench_streaming_perf 2>&1 | tail -20 || echo "No streaming bench target (bench_streaming_perf not in CMakeLists — expected)"
BENCH_BUILD=$?
cd "${SINGLET_ROOT}"

# ---------------------------------------------------------------------------
# Step 5: benchmark-registry.md append.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 5: benchmark-registry.md append ---"

python3 - <<PYEOF
import os, re
from datetime import date

today = date.today().isoformat()
stream_log = "/tmp/cycle66_ctest_streaming.log"

def parse_ctest(log_path):
    n_pass = n_total = n_skip = 0
    wall_sec = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "Test #" in line:
                    n_total += 1
                if "PASSED" in line or "Passed" in line:
                    n_pass += 1
                if "GTEST_SKIP" in line or "SKIPPED" in line:
                    n_skip += 1
                if "Total Test time" in line and "sec" in line:
                    m = re.search(r'([\d.]+)\s+sec', line)
                    if m:
                        wall_sec = float(m.group(1))
    return n_pass, n_total, n_skip, wall_sec

sp, st, ss, sw = parse_ctest(stream_log)
stream_str     = f"{sp}/{st} pass ({ss} GTEST_SKIP)"
stream_wall_str = f"{sw:.2f}s" if sw is not None else "n/a"

build_exit  = "${BUILD_EXIT}"
ctest_exit  = "${STREAM_CTEST}"

row = (
    f"| {today} | streaming/pipeline ctest | full-suite | singlet-gpu | N/A | N/A | N/A | — | — | — "
    f"| ctest: {stream_str}; wall={stream_wall_str}; build={build_exit}; ctest_exit={ctest_exit} | cycle66 |"
)

registry_path = "${STATE_DIR}/benchmark-registry.md"
try:
    with open(registry_path, "a") as f:
        f.write(row + "\n")
    print(f"  benchmark-registry.md: 1 row appended.")
except Exception as e:
    print(f"  WARNING: could not write registry: {e}")

print(f"  Streaming ctest: {stream_str}  wall={stream_wall_str}")
PYEOF

# ---------------------------------------------------------------------------
# Step 6: cycle-log.md episode entry.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: cycle-log.md episode ---"

python3 - <<PYEOF
import os, re
from datetime import datetime, timezone

now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
stream_log = "/tmp/cycle66_ctest_streaming.log"
build_log  = "/tmp/cycle66_build_streaming.log"

def parse_ctest(log_path):
    n_pass = n_total = n_skip = 0
    wall_sec = None
    failed_tests = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                if "Test #" in line:
                    n_total += 1
                if "PASSED" in line or "Passed" in line:
                    n_pass += 1
                if "GTEST_SKIP" in line or "SKIPPED" in line:
                    n_skip += 1
                if "FAILED" in line and "Test #" in line:
                    failed_tests.append(line.strip())
                if "Total Test time" in line and "sec" in line:
                    m = re.search(r'([\d.]+)\s+sec', line)
                    if m:
                        wall_sec = float(m.group(1))
    return n_pass, n_total, n_skip, wall_sec, failed_tests

sp, st, ss, sw, sfailed = parse_ctest(stream_log)
stream_wall_str = f"{sw:.2f}s" if sw is not None else "n/a"

# Extract first 5 unique error lines from build log for the episode.
compile_errors = []
if os.path.exists(build_log):
    seen = set()
    with open(build_log) as f:
        for line in f:
            if "error:" in line.lower() and line not in seen:
                seen.add(line)
                compile_errors.append(line.strip())
                if len(compile_errors) >= 5:
                    break

build_exit  = "${BUILD_EXIT}"
ctest_exit  = "${STREAM_CTEST}"

issues_str = "; ".join(compile_errors[:3]) if compile_errors else ("none" if build_exit == "0" else "see log")

lines = [
    f"\n## Cycle 66 ({now_str}) — Feature 17 Streaming Phase E GPU compile + test\n",
    f"- Build: exit={build_exit}\n",
    f"- ctest: {sp}/{st} passed  ({ss} GTEST_SKIP)  wall={stream_wall_str}  exit={ctest_exit}\n",
    f"- Known disabled tests (CYCLE-7-FOLLOWUP-WRITE-PZ): Pipeline_NmfChunked_RunsToCompletion, Pipeline_GeneMismatch_Errors\n",
    f"- Issues: {issues_str}\n",
    f"- Failed tests: {sfailed if sfailed else 'none'}\n",
    f"- Node: $(hostname)   JobID: ${SLURM_JOB_ID:-local}\n",
    f"- Next: triage compile errors -> DAG fix tasks; re-enable write_pz-gated tests when CYCLE-7-FOLLOWUP-WRITE-PZ resolves\n",
]

cycle_log = "${STATE_DIR}/cycle-log.md"
with open(cycle_log, "a") as f:
    f.writelines(lines)

print(f"cycle-log.md: Cycle 66 episode written.")
PYEOF

# ---------------------------------------------------------------------------
# Step 7: Final summary.
# ---------------------------------------------------------------------------
echo ""
echo "=== Cycle 66 final summary ==="
echo "Date:  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Node:  $(hostname)"
echo "JobID: ${SLURM_JOB_ID:-local}"
echo ""
echo "STREAMING PIPELINE (Feature 17):"
echo "  build exit:    ${BUILD_EXIT}"
echo "  ctest exit:    ${STREAM_CTEST}"
echo "  ctest result:  ${STREAM_PASS:-0}/${STREAM_TOTAL:-0} passed  (${STREAM_SKIP:-0} GTEST_SKIPs)"
echo "  wall time:     ${STREAM_WALL:-n/a}"
echo ""
echo "Known disabled (#if 0):"
echo "  - Pipeline_NmfChunked_RunsToCompletion  (CYCLE-7-FOLLOWUP-WRITE-PZ)"
echo "  - Pipeline_GeneMismatch_Errors           (CYCLE-7-FOLLOWUP-WRITE-PZ)"
echo ""
echo "--- benchmark-registry.md (last 3 rows) ---"
tail -3 "${STATE_DIR}/benchmark-registry.md" || true
echo ""
echo "=== Cycle 66 DONE ==="
