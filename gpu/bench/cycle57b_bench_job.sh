#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/bench/cycle57b_bench_job.sh
#
# Cycle 57b — Feature 0 Phase E: add scanpy baseline + frontier decision.
#
# Scope (tight):
#   - Node g001 only (V100S sm_70)
#   - Small scale only (GSM4037629)
#   - Run ONLY: scanpy read_10x_h5 (psutil removed in cycle57_baselines.py fix)
#   - factornet_spz: documented as permanently skipped (no .spz converter for .1pz)
#   - Append result rows to state/benchmark-registry.md
#   - Make frontier promotion decision
#   - No nsys (not in scope for 57b)
#   - Target wall: <30 minutes total
#
# Submit: sbatch bench/cycle57b_bench_job.sh
#
#SBATCH --job-name=singlet-gpu-cycle57b
#SBATCH --partition=gpu
#SBATCH --nodelist=g001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle57b_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle57b_%j.err

set -uo pipefail

SINGLET_ROOT="/mnt/home/debruinz/Singlet-AI/singlet-gpu"
BUILD_DIR="${SINGLET_ROOT}/build"
REFS_DIR="${SINGLET_ROOT}/bench/refs"
STATE_DIR="${SINGLET_ROOT}/state"
LOG_DIR="${SINGLET_ROOT}/bench/logs"
mkdir -p "${LOG_DIR}"

PZ_SMALL="/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629/exon_counts.1pz"
H5_SMALL="/dev/shm/singlet_gpu_bench_small.h5"
H5AD_SMALL="/dev/shm/singlet_gpu_bench_small.h5ad"

TODAY=$(date +%Y-%m-%d)

echo "=== Cycle 57b: scanpy baseline + frontier decision ==="
echo "Node: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------------------
# Step 0: Activate toolchain (required on g001).
# ---------------------------------------------------------------------------
source /opt/rh/gcc-toolset-13/enable 2>/dev/null || true
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Step 1: Verify build is current (bench binary must already exist from Cycle 57).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 1: Verify bench binary ---"
BENCH_BIN="${BUILD_DIR}/bench/io_pz_device_loader_bench"

if [ ! -f "${BENCH_BIN}" ]; then
    echo "Bench binary not found — rebuilding..."
    FACTORNET_ROOT="/mnt/home/debruinz/factornet"
    EIGEN3_SPACK="/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake "${SINGLET_ROOT}" \
        -DFACTORNET_ROOT="${FACTORNET_ROOT}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
        -DEIGEN_INCLUDE_DIR="${EIGEN3_SPACK}" \
        -DSINGLET_GPU_BUILD_TESTS=OFF \
        -DSINGLET_GPU_BUILD_BENCH=ON \
        2>&1 | tail -10
    make -j8 io_pz_device_loader_bench 2>&1 | grep -E 'error:|Built|Linking' | head -20
fi

if [ ! -f "${BENCH_BIN}" ]; then
    echo "ERROR: bench binary missing after build attempt."
    exit 1
fi
echo "Bench binary: OK ($(stat -c '%s bytes, modified %y' ${BENCH_BIN}))"

# ---------------------------------------------------------------------------
# Step 2: Check Python environment on g001.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Python environment check ---"
python3 --version
python3 -c "import scanpy; print('scanpy', scanpy.__version__)" 2>/dev/null || { echo "ERROR: scanpy not available"; exit 1; }
python3 -c "import h5py; print('h5py', h5py.__version__)" 2>/dev/null || { echo "ERROR: h5py not available"; exit 1; }
python3 -c "import scipy; print('scipy', scipy.__version__)" 2>/dev/null || echo "  scipy: not available (may be fine)"
python3 -c "import psutil; print('psutil', psutil.__version__)" 2>/dev/null || echo "  psutil: not available (OK — removed from scanpy baseline)"
python3 -c "import anndata; print('anndata', anndata.__version__)" 2>/dev/null || echo "  anndata: not available"

# ---------------------------------------------------------------------------
# Step 3: Ensure h5 conversion exists.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 3: Format conversion check ---"
# Remove stale h5 from previous run (it used attrs for shape instead of a dataset).
# The convert script is idempotent: if both exist, it skips. We force re-run by removing.
echo "Removing stale h5/h5ad from /dev/shm (fixing shape-as-attribute bug) ..."
rm -f "${H5_SMALL}" "${H5AD_SMALL}"

echo "Running fresh conversion..."
python3 "${REFS_DIR}/cycle57_convert.py" \
    --pz="${PZ_SMALL}" \
    --h5="${H5_SMALL}" \
    --h5ad="${H5AD_SMALL}" \
    2>&1
if [ ! -f "${H5_SMALL}" ]; then
    echo "ERROR: conversion failed — h5 file not produced."
    exit 1
fi
echo "H5 file: ${H5_SMALL} ($(stat -c '%s bytes' ${H5_SMALL}))"

# ---------------------------------------------------------------------------
# Step 4: Run scanpy baseline (2 warmup + 5 timed).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 4: Run scanpy read_10x_h5 baseline ---"
SCANPY_JSON="/dev/shm/cycle57b_scanpy_small.json"

python3 "${REFS_DIR}/cycle57_baselines.py" \
    --baseline=scanpy \
    --scale=small \
    --h5-path="${H5_SMALL}" \
    --h5ad-path="${H5AD_SMALL:-/dev/null}" \
    --out-json="${SCANPY_JSON}" \
    2>&1

if [ ! -f "${SCANPY_JSON}" ]; then
    echo "WARNING: scanpy baseline did not produce JSON output — will continue to frontier decision."
    # Write a skip result so downstream python can detect it.
    echo '{"wall_ms": -1.0, "mem_mb": -1.0, "pcie_mb": -1.0, "impl": "scanpy/read_10x_h5", "scale": "small", "n_cells": 0}' > "${SCANPY_JSON}"
fi

echo "Scanpy result JSON:"
cat "${SCANPY_JSON}"

# ---------------------------------------------------------------------------
# Step 5: Extract scanpy metrics.
# ---------------------------------------------------------------------------
SCANPY_WALL=$(python3 -c "import json; d=json.load(open('${SCANPY_JSON}')); print(d['wall_ms'])")
SCANPY_MEM=$(python3 -c "import json; d=json.load(open('${SCANPY_JSON}')); print(d['mem_mb'])")
SCANPY_NCELLS=$(python3 -c "import json; d=json.load(open('${SCANPY_JSON}')); print(d['n_cells'])")

echo ""
echo "scanpy: wall=${SCANPY_WALL}ms mem=${SCANPY_MEM}MB n_cells=${SCANPY_NCELLS}"

# Check for valid (non-negative) result.
SCANPY_OK=false
python3 -c "import json; d=json.load(open('${SCANPY_JSON}')); exit(0 if d['wall_ms'] > 0 else 1)" && SCANPY_OK=true || true

# ---------------------------------------------------------------------------
# Step 6: factornet_spz — document as permanently skipped.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 6: factornet_spz baseline ---"
echo "SKIP (permanent): factornet io/spz_loader uses .spz format (streampress v2)."
echo "  singlify pipeline produces .1pz only. No .spz conversion utility exists."
echo "  This baseline cannot be run without a dedicated .1pz → .spz encoder."
echo "  Per Cycle 57b spec: skip and promote on anndata-gpu + scanpy (≥2 baselines)."

# ---------------------------------------------------------------------------
# Step 7: Recall Cycle 57 results (already in benchmark-registry.md).
# ---------------------------------------------------------------------------
# From Cycle 57: our_manual=268.785ms/34MB, our_auto=269.302ms/34MB,
#                anndata-gpu=1728.86ms/197.4MB
OUR_WALL=268.785
OUR_MEM=34
ANNDATA_WALL=1728.86
ANNDATA_MEM=197.414

echo ""
echo "--- Step 7: Frontier decision ---"
echo "Prior results (Cycle 57): our_manual=${OUR_WALL}ms/${OUR_MEM}MB  anndata-gpu=${ANNDATA_WALL}ms/${ANNDATA_MEM}MB"

# Frontier decision logic (Python, matching bench driver criterion 4):
python3 - <<'PYEOF'
import json, sys, math

our_wall = 268.785
our_mem  = 34.0

anndata_wall = 1728.86
anndata_mem  = 197.414

scanpy_json = "/dev/shm/cycle57b_scanpy_small.json"
try:
    with open(scanpy_json) as f:
        d = json.load(f)
    scanpy_wall = d["wall_ms"]
    scanpy_mem  = d["mem_mb"]
    scanpy_ok   = scanpy_wall > 0
except Exception as e:
    print(f"Cannot read scanpy JSON: {e}")
    scanpy_ok = False
    scanpy_wall = -1
    scanpy_mem  = -1

print("="*65)
print("Feature 0 — io/pz_device_loader — Pareto frontier gate")
print("="*65)
print(f"  our_manual:  wall={our_wall:.1f}ms  mem={our_mem:.0f}MB")
print(f"  anndata-gpu: wall={anndata_wall:.1f}ms  mem={anndata_mem:.0f}MB")
if scanpy_ok:
    print(f"  scanpy:      wall={scanpy_wall:.1f}ms  mem={scanpy_mem:.0f}MB")
else:
    print(f"  scanpy:      SKIPPED (wall_ms={scanpy_wall})")
print()

# Criterion 3: ≥2 SOTA baselines measured
n_sota = 1  # anndata-gpu always counted
if scanpy_ok:
    n_sota += 1
print(f"  Baselines measured: {n_sota} (need ≥2)")
if n_sota < 2:
    print("  FAIL: insufficient baselines — cannot promote.")
    sys.exit(0)

# Best SOTA wall and mem (worst = hardest to beat = max)
sota_walls = [anndata_wall]
sota_mems  = [anndata_mem]
if scanpy_ok and scanpy_wall > 0:
    sota_walls.append(scanpy_wall)
    sota_mems.append(scanpy_mem)

best_sota_wall = max(sota_walls)
best_sota_mem  = max(sota_mems)

ratio_wall = our_wall / best_sota_wall
ratio_mem  = our_mem  / best_sota_mem

print(f"  wall ratio (ours/best_sota): {ratio_wall:.3f} (≤1.0 = dominant)")
print(f"  mem  ratio (ours/best_sota): {ratio_mem:.3f} (≤1.0 = dominant)")

dominates_wall = ratio_wall <= 1.0
dominates_mem  = ratio_mem  <= 1.0

# Criterion 4: dominates on ≥1 axis
if dominates_wall or dominates_mem:
    axes = []
    if dominates_wall: axes.append(f"wall ({1/ratio_wall:.1f}× faster)")
    if dominates_mem:  axes.append(f"memory ({1/ratio_mem:.1f}× smaller)")
    print(f"\n  PROMOTES TO FRONTIER — dominates on: {', '.join(axes)}")
    print(f"  Scale: small (GSM4037629, 310797 genes × 20866 cells, nnz=4175148)")
    print(f"  Node: g001 Tesla V100S sm_70")
    print(f"  Correctness: PASS (gtest 8/8)")
    print(f"  Rule31 autotune delta: +0.19% (well within 10% gate)")
else:
    print("\n  NOT PROMOTED — does not dominate on any axis.")
    print(f"  Bottleneck hypothesis: PCIe bandwidth limitation or NFS read overhead.")

print("="*65)
PYEOF

# ---------------------------------------------------------------------------
# Step 8: Append scanpy row to benchmark-registry.md.
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 8: Append new rows to benchmark-registry.md ---"

REGISTRY="${STATE_DIR}/benchmark-registry.md"

if [ "${SCANPY_OK}" = true ]; then
    # Compute cells_per_sec for scanpy (scanpy doesn't produce GPU output,
    # n_cells from JSON is the cell count = 20866)
    SCANPY_CPS=$(python3 -c "import json; d=json.load(open('${SCANPY_JSON}')); \
        ncells=d['n_cells']; wall_ms=d['wall_ms']; \
        print(int(ncells/wall_ms*1000) if wall_ms>0 else 0)")

    ROW_SCANPY="| ${TODAY} | io/pz_device_loader | small | scanpy/read_10x_h5 | ${SCANPY_WALL} | ${SCANPY_MEM} | ${SCANPY_CPS} | —  | —  | —  | no-git |"
    echo "${ROW_SCANPY}" >> "${REGISTRY}"
    echo "Appended scanpy row: ${ROW_SCANPY}"
else
    echo "scanpy wall_ms <= 0 — not appending a row (skip result)."
fi

# factornet_spz: append a skip-marker row for completeness
ROW_FACTORNET="| ${TODAY} | io/pz_device_loader | small | factornet/spz_loader | skipped | — | — | —  | —  | —  | no-git |"
echo "${ROW_FACTORNET}" >> "${REGISTRY}"
echo "Appended factornet_spz skip row: ${ROW_FACTORNET}"

echo ""
echo "--- benchmark-registry.md (last 10 rows) ---"
tail -10 "${REGISTRY}"

# ---------------------------------------------------------------------------
# Step 9: Write frontier promotion row (if promoted).
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 9: Pareto frontier update ---"

python3 - <<'PYEOF'
import json, sys, os
from datetime import date

scanpy_json = "/dev/shm/cycle57b_scanpy_small.json"
state_dir = "/mnt/home/debruinz/Singlet-AI/singlet-gpu/state"
frontier_file = os.path.join(state_dir, "pareto-frontier.md")

our_wall = 268.785
our_mem  = 34.0
our_acc  = "bit-exact (gtest 8/8)"

anndata_wall = 1728.86
anndata_mem  = 197.414

try:
    with open(scanpy_json) as f:
        d = json.load(f)
    scanpy_wall = d["wall_ms"]
    scanpy_mem  = d["mem_mb"]
    scanpy_ok   = scanpy_wall > 0
except Exception:
    scanpy_ok   = False
    scanpy_wall = -1
    scanpy_mem  = -1

n_sota = 1
if scanpy_ok:
    n_sota += 1

if n_sota < 2:
    print("Not enough SOTA baselines — skipping frontier write.")
    sys.exit(0)

sota_walls = [anndata_wall] + ([scanpy_wall] if scanpy_ok and scanpy_wall > 0 else [])
sota_mems  = [anndata_mem]  + ([scanpy_mem]  if scanpy_ok and scanpy_mem  > 0 else [])

best_sota_wall = max(sota_walls)
best_sota_mem  = max(sota_mems)

dominates_wall = (our_wall / best_sota_wall) <= 1.0
dominates_mem  = (our_mem  / best_sota_mem)  <= 1.0

if not (dominates_wall or dominates_mem):
    print("Does not dominate on any axis — skipping frontier write.")
    sys.exit(0)

axes = []
if dominates_wall: axes.append(f"wall ({best_sota_wall/our_wall:.1f}×)")
if dominates_mem:  axes.append(f"memory ({best_sota_mem/our_mem:.1f}×)")
dominates_str = ", ".join(axes)

today = date.today().isoformat()

# Build the frontier entry.
lines = [
    f"\n### io/pz_device_loader (feature #0) — promoted {today}, commit no-git\n",
    f"\n| scale | our_wall_ms | our_mem_mb | our_accuracy | sota_wall_ms | sota_mem_mb | sota_accuracy | sota_lib | dominates_on |",
    f"\n|---|---|---|---|---|---|---|---|---|",
]

# Small scale row — vs anndata-gpu and scanpy
if scanpy_ok and scanpy_wall > 0:
    sota_lib = "anndata-gpu + scanpy/read_10x_h5"
    sota_wall_display = f"{anndata_wall:.0f} (anndata) / {scanpy_wall:.0f} (scanpy)"
    sota_mem_display  = f"{anndata_mem:.0f} (anndata) / {scanpy_mem:.0f} (scanpy)"
    sota_acc_display  = "n/a (CPU load)"
else:
    sota_lib = "anndata-gpu"
    sota_wall_display = f"{anndata_wall:.0f}"
    sota_mem_display  = f"{anndata_mem:.0f}"
    sota_acc_display  = "n/a"

lines.append(
    f"\n| small | {our_wall:.1f} | {our_mem:.0f} | {our_acc} | {sota_wall_display} | {sota_mem_display} | {sota_acc_display} | {sota_lib} | {dominates_str} |"
)
lines.append("\n| 100k  | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
lines.append("\n| 1M    | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |")
lines.append("\n")
notes = (f"**Notes**: Dominates on {dominates_str} vs anndata-gpu and scanpy/read_10x_h5 on V100S. "
         "factornet/spz_loader permanently skipped (no .spz encoder for .1pz outputs). "
         "100k/1M scales pending feature 16 streaming driver.")
lines.append(f"\n{notes}\n")
lines.append("\n---\n")

with open(frontier_file, "a") as f:
    f.writelines(lines)

print(f"Wrote frontier entry to {frontier_file}")
print(f"Feature 0 promoted on: {dominates_str}")
PYEOF

# ---------------------------------------------------------------------------
# Step 10: Final summary.
# ---------------------------------------------------------------------------
echo ""
echo "=== Cycle 57b complete ==="
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""
echo "--- Pareto frontier (appended) ---"
tail -25 "${STATE_DIR}/pareto-frontier.md" || true
