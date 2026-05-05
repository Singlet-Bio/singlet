#!/usr/bin/env bash
# ── Staggered quant launcher ─────────────────────────────────────────
# Scans $SCRATCH/scgeo_downloads for completed downloads that haven't
# been quantified yet, then submits quant jobs to process them.
#
# Each quant job scans and claims samples via O_EXCL lock files,
# so over-submitting is safe (excess jobs exit cleanly).
#
# Usage:
#   bash launch_quant.sh              # auto-detect ready count, 2s apart
#   bash launch_quant.sh 20 2 10      # 20 jobs, 2s apart, 10 samples/job
#   bash launch_quant.sh 0            # dry-run: just count ready samples
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"
DL_DIR="$SCRATCH/scgeo_downloads"

DELAY_SECS="${2:-2}"
QUANT_BATCH="${3:-10}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QT_SCRIPT="$SCRIPT_DIR/anvil_quant.sh"

if [[ ! -f "$QT_SCRIPT" ]]; then
    echo "ERROR: Cannot find $QT_SCRIPT"
    exit 1
fi

# ── Count ready samples ───────────────────────────────────────────────
echo "Scanning $DL_DIR for ready downloads..."
READY=$(find "$DL_DIR" -name "download_manifest.json" -exec sh -c '
    dir=$(dirname "$1")
    if [ -f "$dir/.quant_done" ]; then exit 1; fi
    if [ -f "$dir/.quant_locked" ]; then exit 1; fi
    python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get(\"success\") else 1)" "$1"
' _ {} \; -print 2>/dev/null | wc -l)

echo "Ready samples: $READY"

if [[ "$READY" -eq 0 ]]; then
    echo "No ready samples found — nothing to submit"
    exit 0
fi

# Auto-detect N_JOBS or use first arg
if [[ -n "${1:-}" ]]; then
    N_JOBS="$1"
else
    # Submit enough jobs to cover ready samples, with headroom
    N_JOBS=$(( (READY + QUANT_BATCH - 1) / QUANT_BATCH ))
    # Cap at 50 to avoid flooding the queue
    if [[ $N_JOBS -gt 50 ]]; then
        N_JOBS=50
    fi
fi

if [[ "$N_JOBS" -eq 0 ]]; then
    echo "(dry-run mode — not submitting jobs)"
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "Launching $N_JOBS quant jobs"
echo "  Ready:      $READY samples"
echo "  Batch size: $QUANT_BATCH samples/job"
echo "  Delay:      ${DELAY_SECS}s between submissions"
echo "  SU cost:    ~8 SU/hr/job (4 CPU, 16GB)"
echo "════════════════════════════════════════════════════"
echo ""

SUBMITTED=0
FAILED=0

for i in $(seq 1 "$N_JOBS"); do
    JOB_ID=$(sbatch \
        --export=ALL,QUANT_BATCH="$QUANT_BATCH" \
        --parsable \
        "$QT_SCRIPT" 2>&1) || true

    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        SUBMITTED=$((SUBMITTED + 1))
        echo "[$i/$N_JOBS] Submitted job $JOB_ID (batch=$QUANT_BATCH)"
    else
        FAILED=$((FAILED + 1))
        echo "[$i/$N_JOBS] FAILED: $JOB_ID"
        if [[ $FAILED -ge 3 ]]; then
            echo "3 consecutive failures — stopping submissions"
            break
        fi
    fi

    if [[ $i -lt $N_JOBS ]]; then
        sleep "$DELAY_SECS"
    fi
done

echo ""
echo "════════════════════════════════════════════════════"
echo "Submitted: $SUBMITTED / $N_JOBS"
echo "Failed:    $FAILED"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER -n scgeo-qt"
echo "  find \$SCRATCH/scgeo_downloads -name '.quant_done' | wc -l"
echo "  ls \$SCGEO_BASE/pipeline/results/results_quant_*.csv | wc -l"
echo "════════════════════════════════════════════════════"
