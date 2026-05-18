#!/usr/bin/env bash
# singlet-gpu benchmark runner
# Usage: bash bench/run_all.sh [--build-dir <dir>] [--scales tiny,10k,100k]
# Idempotent: safe to run multiple times; exits 0 without GPU or missing binaries.
# Output: appends rows to state/benchmark-registry.md; logs stderr per feature.
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
REGISTRY="${REPO_ROOT}/state/benchmark-registry.md"
LOG_DIR="${SCRIPT_DIR}/logs"
QUANT_BASE="/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna"
SAMPLE_10K="${QUANT_BASE}/GSE127/GSE127918/GSM4037629"
SAMPLE_CONCAT="${QUANT_BASE}"  # driver scans for 100k concat if present
SCALES="${BENCH_SCALES:-tiny,10k}"  # override with BENCH_SCALES=tiny,10k,100k

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --scales)    SCALES="$2";    shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Timestamp + commit
# ---------------------------------------------------------------------------
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
if git -C "${REPO_ROOT}" rev-parse HEAD &>/dev/null 2>&1; then
  COMMIT=$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "no-git")
else
  COMMIT="no-git"
fi

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Guard: GPU required
# ---------------------------------------------------------------------------
if ! nvidia-smi &>/dev/null 2>&1; then
  echo "NO GPU — skipping bench (nvidia-smi not found or failed)"
  exit 0
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
echo "GPU detected: ${GPU_NAME}"

# ---------------------------------------------------------------------------
# Guard: build directory
# ---------------------------------------------------------------------------
BENCH_BIN_DIR="${BUILD_DIR}/bench"
if [[ ! -d "${BENCH_BIN_DIR}" ]]; then
  echo "Build directory ${BENCH_BIN_DIR} not found — run cmake + make first."
  echo "Hint: cmake -S ${REPO_ROOT} -B ${BUILD_DIR} && cmake --build ${BUILD_DIR} -j"
  exit 0
fi

# Collect bench binaries
mapfile -t BENCH_BINARIES < <(find "${BENCH_BIN_DIR}" -maxdepth 1 -type f -name "bench_*_perf" -executable 2>/dev/null | sort)
if [[ ${#BENCH_BINARIES[@]} -eq 0 ]]; then
  echo "No bench_*_perf binaries found in ${BENCH_BIN_DIR} — nothing to run."
  exit 0
fi
echo "Found ${#BENCH_BINARIES[@]} bench driver(s): $(basename -a "${BENCH_BINARIES[@]}" | tr '\n' ' ')"

# ---------------------------------------------------------------------------
# Registry header guard
# ---------------------------------------------------------------------------
ensure_registry_header() {
  local header="| feature | scale | wall_ms_min | wall_ms_med | wall_ms_max | mem_mb_peak | cells_per_s | sota_wall_sec | sota_mem_mb | ratio_wall | commit | timestamp |"
  local sep="|---|---|---|---|---|---|---|---|---|---|---|---|"
  if [[ ! -f "${REGISTRY}" ]]; then
    {
      echo "# singlet-gpu — Benchmark Registry"
      echo ""
      echo "Append-only. One row per {feature, scale} measurement. Written by run_all.sh."
      echo ""
      echo "${header}"
      echo "${sep}"
    } > "${REGISTRY}"
    echo "Created ${REGISTRY} with header."
  elif ! grep -q "| feature |" "${REGISTRY}"; then
    # File exists but has no table header — prepend
    TMP=$(mktemp)
    {
      echo "${header}"
      echo "${sep}"
      cat "${REGISTRY}"
    } > "${TMP}"
    mv "${TMP}" "${REGISTRY}"
    echo "Prepended missing table header to ${REGISTRY}."
  fi
}

# ---------------------------------------------------------------------------
# Run a single binary at a given scale
# ---------------------------------------------------------------------------
run_driver() {
  local bin="$1"
  local scale="$2"
  local feature
  feature=$(basename "${bin}" | sed 's/^bench_//; s/_perf$//')

  local log_file="${LOG_DIR}/${TIMESTAMP}_${feature}_${scale}.log"
  echo "  Running ${feature} @ ${scale} ..."

  # Scale-specific arguments
  local scale_args=""
  case "${scale}" in
    tiny)  scale_args="--scale tiny" ;;
    10k)
      if [[ ! -d "${SAMPLE_10K}" ]]; then
        echo "  SKIP ${feature}@${scale}: sample not found at ${SAMPLE_10K}"
        return 0
      fi
      scale_args="--scale 10k --sample-dir ${SAMPLE_10K}"
      ;;
    100k)
      scale_args="--scale 100k --sample-base ${SAMPLE_CONCAT}"
      ;;
    1m)
      # 1M streaming: deferred to dedicated streaming driver
      echo "  SKIP ${feature}@${scale}: 1M streaming deferred (use bench_streaming_perf)"
      return 0
      ;;
    *)
      echo "  SKIP unknown scale: ${scale}"
      return 0
      ;;
  esac

  # Run the driver; capture exit code without aborting the script
  local exit_code=0
  "${bin}" ${scale_args} \
    --commit "${COMMIT}" \
    --timestamp "${TIMESTAMP}" \
    --registry "${REGISTRY}" \
    2>"${log_file}" || exit_code=$?

  if [[ ${exit_code} -ne 0 ]]; then
    echo "  WARNING: ${feature}@${scale} exited ${exit_code} — see ${log_file}"
  else
    echo "  OK: ${feature}@${scale} — log: ${log_file}"
  fi
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
ensure_registry_header

IFS=',' read -ra SCALE_LIST <<< "${SCALES}"

echo ""
echo "=== singlet-gpu bench run ==="
echo "  commit    : ${COMMIT}"
echo "  timestamp : ${TIMESTAMP}"
echo "  gpu       : ${GPU_NAME}"
echo "  scales    : ${SCALES}"
echo "  registry  : ${REGISTRY}"
echo ""

for bin in "${BENCH_BINARIES[@]}"; do
  feature=$(basename "${bin}" | sed 's/^bench_//; s/_perf$//')
  echo "--- ${feature} ---"
  for scale in "${SCALE_LIST[@]}"; do
    run_driver "${bin}" "${scale}"
  done
done

echo ""
echo "=== bench run complete — results appended to ${REGISTRY} ==="
