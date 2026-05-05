#!/usr/bin/env bash
# ── Staggered download launcher ──────────────────────────────────────
# Submits N download jobs with a delay between each to avoid SRA/ENA
# rate limiting. Each job claims its own batch via grab_batch.py.
#
# Usage:
#   bash launch_downloads.sh              # 50 jobs, 5s apart, phase 4a
#   bash launch_downloads.sh 100 3 4a     # 100 jobs, 3s apart, phase 4a
#   bash launch_downloads.sh 20 10 1      # 20 jobs, 10s apart, phase 1
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

N_JOBS="${1:-50}"
DELAY_SECS="${2:-5}"
PHASE="${3:-4a}"
BATCH_SIZE="${4:-20}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DL_SCRIPT="$SCRIPT_DIR/anvil_download.sh"

if [[ ! -f "$DL_SCRIPT" ]]; then
    echo "ERROR: Cannot find $DL_SCRIPT"
    exit 1
fi

echo "════════════════════════════════════════════════════"
echo "Launching $N_JOBS download jobs"
echo "  Phase:      $PHASE"
echo "  Batch size: $BATCH_SIZE samples/job"
echo "  Delay:      ${DELAY_SECS}s between submissions"
echo "  Total:      ~$((N_JOBS * BATCH_SIZE)) samples"
echo "  SU cost:    ~4 SU/hr/job (1 CPU, 8GB)"
echo "════════════════════════════════════════════════════"
echo ""

SUBMITTED=0
FAILED=0

for i in $(seq 1 "$N_JOBS"); do
    JOB_ID=$(sbatch \
        --export=ALL,PHASE="$PHASE",BATCH_SIZE="$BATCH_SIZE" \
        --parsable \
        "$DL_SCRIPT" 2>&1) || true

    if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
        SUBMITTED=$((SUBMITTED + 1))
        echo "[$i/$N_JOBS] Submitted job $JOB_ID (phase=$PHASE, batch=$BATCH_SIZE)"
    else
        FAILED=$((FAILED + 1))
        echo "[$i/$N_JOBS] FAILED: $JOB_ID"
        # If submission fails, likely a queue limit — stop early
        if [[ $FAILED -ge 3 ]]; then
            echo "3 consecutive failures — stopping submissions"
            break
        fi
    fi

    # Stagger to avoid rate limiting
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
echo "  squeue -u \$USER -n scgeo-dl"
echo "  ls \$SCRATCH/scgeo_downloads/ | wc -l"
echo "  du -sh \$SCRATCH/scgeo_downloads/"
echo "════════════════════════════════════════════════════"
