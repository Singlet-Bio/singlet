#!/bin/bash
#
# Anvil Launch — Submit Phase 4a (Mouse) array job.
#
# Run AFTER:
#   1. Bootstrap completed (01_bootstrap.sh) — env + index ready
#   2. Globus verified (02_globus_setup.sh) — can transfer results back
#   3. Smoke test passed (03_smoke_test.sh / smoke_test.sh)
#
# This script submits the full array job for 21,304 eligible mouse samples.
# Each task claims a batch of 10 samples via the atomic grab_batch system.
#
# Resubmit after all tasks finish to drain remaining unclaimed samples.
#
# Usage:
#   export ALLOCATION=cis250209
#   bash 04_launch.sh [--phase 4a|4b] [--array-size 500] [--batch-size 10]

set -euo pipefail

ALLOCATION="${ALLOCATION:-YOUR_ALLOCATION}"
PHASE="${1:---phase}"
shift || true
PHASE="${1:-4a}"
shift 2>/dev/null || true

# Parse optional args
ARRAY_SIZE=500
BATCH_SIZE=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --array-size)  ARRAY_SIZE="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --phase)       PHASE="$2"; shift 2 ;;
        *)             echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$ALLOCATION" == "YOUR_ALLOCATION" ]]; then
    echo "ERROR: Set ALLOCATION first:"
    echo "  export ALLOCATION=cis250209"
    exit 1
fi

export SCGEO_BASE="$PROJECT/scgeo"

echo "════════════════════════════════════════════════════"
echo "  Anvil Launch — Phase $PHASE"
echo "  Allocation: $ALLOCATION"
echo "  Array size: $ARRAY_SIZE tasks"
echo "  Batch size: $BATCH_SIZE samples/task"
echo "  Max samples: $((ARRAY_SIZE * BATCH_SIZE))"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════"

# Preflight checks
echo ""
echo "▸ Preflight checks..."

# Check index
if [[ ! -d "$SCGEO_BASE/index/mouse_splici" ]]; then
    echo "  ERROR: Mouse index not found. Run 01_bootstrap.sh first."
    exit 1
fi
echo "  Index: $(du -sh "$SCGEO_BASE/index/mouse_splici" | cut -f1)"

# Check catalog
if [[ ! -f "$SCGEO_BASE/catalog/processing_catalog.parquet" ]]; then
    echo "  ERROR: Catalog not found. Transfer from Clipper first."
    exit 1
fi
echo "  Catalog: $(ls -lh "$SCGEO_BASE/catalog/processing_catalog.parquet" | awk '{print $5}')"

# Check claims dir
mkdir -p "$SCGEO_BASE/pipeline/claims"
mkdir -p "$SCGEO_BASE/pipeline/results"
mkdir -p "$SCGEO_BASE/pipeline/logs"
echo "  Claims dir: OK"

# Check balance
echo ""
echo "▸ SU balance:"
mybalance 2>/dev/null || echo "  (mybalance not available — check via ACCESS portal)"

# Submit
echo ""
echo "▸ Submitting Phase $PHASE array..."

WORKER="$(dirname "$0")/anvil_worker.sh"
if [[ ! -f "$WORKER" ]]; then
    echo "  ERROR: Worker script not found: $WORKER"
    exit 1
fi

# Anvil shared partition: fractional billing by max(cores/128, mem/257GB)
# 8 CPUs + 64GB = max(0.0625, 0.249) × 128 = 32 SU/hr per task
# No concurrent task limit — Anvil scheduler handles node packing
ARRAY_SPEC="0-$((ARRAY_SIZE - 1))"

JOB=$(sbatch -A "$ALLOCATION" \
    --export="ALL,PHASE=$PHASE,BATCH_SIZE=$BATCH_SIZE" \
    --array="$ARRAY_SPEC" \
    --output="$SCGEO_BASE/pipeline/logs/p${PHASE}_%A_%a.out" \
    --error="$SCGEO_BASE/pipeline/logs/p${PHASE}_%A_%a.err" \
    "$WORKER" 2>&1 | grep -oP '\d+')

echo "  Job ID: $JOB"
echo "  Array: $ARRAY_SPEC ($ARRAY_SIZE tasks)"
echo "  Worker: $WORKER"
echo ""
echo "▸ Monitoring:"
echo "  squeue -u \$USER                     # All your jobs"
echo "  squeue -j $JOB                       # This array"
echo "  tail -f $SCGEO_BASE/pipeline/logs/p${PHASE}_${JOB}_0.out  # First task"
echo ""
echo "▸ SU cost estimate:"
echo "  At 32 SU/hr per task, ~1hr per 10-sample batch:"
echo "  $((ARRAY_SIZE * 32)) SU max (if all tasks run full 12hr)"
echo "  ~$((ARRAY_SIZE * BATCH_SIZE * 3)) SU expected (at ~3 SU/sample)"
echo ""
echo "▸ After all tasks finish, resubmit to drain remaining:"
echo "  bash 04_launch.sh --phase $PHASE --array-size 200"
echo ""
echo "▸ Transfer results back to Clipper:"
echo "  globus transfer \$ANVIL_EP:\$PROJECT/scgeo/pipeline/quant/ \\"
echo "    \$CLIPPER_EP:/mnt/projects/debruinz_project/cellarium/pipeline/quant/ \\"
echo "    --recursive --label 'Phase $PHASE results' --sync-level checksum"
