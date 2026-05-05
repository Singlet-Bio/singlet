#!/usr/bin/env bash
#SBATCH --job-name=scgeo-orch
#SBATCH --account=bio260157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/anvil/projects/x-bio260157/scgeo/pipeline/logs/orch_%j.out
#
# ── Autonomous Pipeline Orchestrator ─────────────────────────────────
# Self-resubmitting SLURM job that monitors the download→quant pipeline
# and keeps both phases saturated. Runs every POLL_INTERVAL seconds
# for up to 4 hours, then resubmits itself.
#
# Backoff triggers:
#   - Scratch usage > SCRATCH_MAX_GB  → pause downloads, only run quant
#   - Scratch usage > SCRATCH_CRIT_GB → pause everything, wait for cleanup
#   - sbatch failures                 → exponential backoff
#   - No eligible samples remaining   → exit permanently
#
# SU cost: 1 SU/hr (1 CPU, 2GB → billing=1 core on Anvil shared)
#
# Usage:
#   sbatch orchestrator.sh                    # default settings
#   POLL_INTERVAL=120 sbatch orchestrator.sh  # check every 2 min
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"

# ── Tuning parameters ────────────────────────────────────────────────
POLL_INTERVAL="${POLL_INTERVAL:-180}"       # seconds between checks
MAX_DL_JOBS="${MAX_DL_JOBS:-25}"           # max concurrent download jobs (was 50; reduced to ease ENA rate-limiting)
MAX_QT_JOBS="${MAX_QT_JOBS:-50}"            # max concurrent quant jobs
DL_BATCH_SIZE="${DL_BATCH_SIZE:-20}"        # samples per download job
QT_BATCH_SIZE="${QT_BATCH_SIZE:-5}"         # samples per quant job
PHASE="${PHASE:-4a}"
SCRATCH_MAX_GB="${SCRATCH_MAX_GB:-75000}"    # pause downloads above this (75TB)
SCRATCH_CRIT_GB="${SCRATCH_CRIT_GB:-90000}"  # pause everything above this (90TB)
DL_SUBMIT_DELAY="${DL_SUBMIT_DELAY:-3}"     # seconds between dl submissions
QT_SUBMIT_DELAY="${QT_SUBMIT_DELAY:-1}"     # seconds between qt submissions
RESUBMIT="${RESUBMIT:-true}"                # resubmit self when time runs out

SCRIPT_DIR="$PROJECT/geo-reprocess/scripts/anvil"
DL_SCRIPT="$SCRIPT_DIR/anvil_download.sh"
QT_SCRIPT="$SCRIPT_DIR/anvil_quant.sh"
XL_SCRIPT="$SCRIPT_DIR/anvil_quant_xl.sh"
ORCH_SCRIPT="$SCRIPT_DIR/orchestrator.sh"
DL_DIR="$SCRATCH/scgeo_downloads"
LOG_DIR="$PROJECT/scgeo/pipeline/logs"

REPORT_SCRIPT="$SCRIPT_DIR/pipeline_report.sh"
REPORT_EMAIL="${REPORT_EMAIL:-debruinz@gvsu.edu}"
EMAIL_INTERVAL="${EMAIL_INTERVAL:-3600}"  # seconds between emails (1hr)

mkdir -p "$LOG_DIR"

# Source the report generator
if [[ -f "$REPORT_SCRIPT" ]]; then
    source "$REPORT_SCRIPT"
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline Orchestrator | job ${SLURM_JOB_ID:-local}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Poll: ${POLL_INTERVAL}s | DL cap: $MAX_DL_JOBS | QT cap: $MAX_QT_JOBS"
echo "  Scratch limits: ${SCRATCH_MAX_GB}GB soft / ${SCRATCH_CRIT_GB}GB hard"
echo "  Script dir: $SCRIPT_DIR"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Helper: count running jobs by name ────────────────────────────────
count_jobs() {
    local name="$1"
    squeue -u "$USER" -n "$name" -h -t RUNNING,PENDING 2>/dev/null | wc -l
}

# ── Helper: get scratch usage in GB ───────────────────────────────────
scratch_gb() {
    du -s --block-size=1G "$DL_DIR" 2>/dev/null | cut -f1
}

# ── Helper: count ready-for-quant samples (fast: no python) ──────────
count_ready() {
    local count=0
    while IFS= read -r mpath; do
        dir=$(dirname "$mpath")
        [[ -f "$dir/.quant_done" ]] && continue
        [[ -f "$dir/.quant_locked" ]] && continue
        grep -q '"success": true' "$mpath" 2>/dev/null && count=$((count + 1))
    done < <(find "$DL_DIR" -name "download_manifest.json" 2>/dev/null)
    echo "$count"
}

# ── Helper: count successful downloads total ─────────────────────────
count_success_dl() {
    find "$DL_DIR" -name "download_manifest.json" \
        -exec grep -l '"success": true' {} + 2>/dev/null | wc -l
}

# ── Helper: count quant done ─────────────────────────────────────────
count_quant_done() {
    find "$DL_DIR" -name ".quant_done" 2>/dev/null | wc -l
}

# ── Helper: submit N download jobs ───────────────────────────────────
submit_downloads() {
    local n="$1"
    local submitted=0 failed=0
    for i in $(seq 1 "$n"); do
        JOB_ID=$(sbatch \
            --export=ALL,PHASE="$PHASE",BATCH_SIZE="$DL_BATCH_SIZE" \
            --parsable \
            "$DL_SCRIPT" 2>&1) || true
        if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            submitted=$((submitted + 1))
        else
            echo "    [ERR] dl sbatch: $JOB_ID"
            failed=$((failed + 1))
            if [[ $failed -ge 3 ]]; then
                echo "    [WARN] 3 dl submit failures — stopping this round"
                break
            fi
        fi
        sleep "$DL_SUBMIT_DELAY"
    done
    echo "    DL submitted: $submitted (failed: $failed)"
}

# ── Helper: submit N quant jobs ──────────────────────────────────────
submit_quants() {
    local n="$1"
    local submitted=0 failed=0
    for i in $(seq 1 "$n"); do
        JOB_ID=$(sbatch \
            --export=ALL,QUANT_BATCH="$QT_BATCH_SIZE" \
            --parsable \
            "$QT_SCRIPT" 2>&1) || true
        if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            submitted=$((submitted + 1))
        else
            echo "    [ERR] qt sbatch: $JOB_ID"
            failed=$((failed + 1))
            if [[ $failed -ge 3 ]]; then
                echo "    [WARN] 3 qt submit failures — stopping this round"
                break
            fi
        fi
        sleep "$QT_SUBMIT_DELAY"
    done
    echo "    QT submitted: $submitted (failed: $failed)"
}

# ── Helper: submit XL quant jobs for .needs_xl samples ───────────────
submit_xl_quants() {
    local n="$1"
    local submitted=0 failed=0
    for i in $(seq 1 "$n"); do
        JOB_ID=$(sbatch \
            --export=ALL,QUANT_BATCH=1 \
            --parsable \
            "$XL_SCRIPT" 2>&1) || true
        if [[ "$JOB_ID" =~ ^[0-9]+$ ]]; then
            submitted=$((submitted + 1))
        else
            echo "    [ERR] xl sbatch: $JOB_ID"
            failed=$((failed + 1))
            if [[ $failed -ge 3 ]]; then break; fi
        fi
        sleep "$QT_SUBMIT_DELAY"
    done
    echo "    XL submitted: $submitted (failed: $failed)"
}

# ── Helper: count .needs_xl samples ───────────────────────────────────
count_needs_xl() {
    find "$DL_DIR" -name ".needs_xl" 2>/dev/null | wc -l
}

# ── Main loop ─────────────────────────────────────────────────────────
CYCLE=0
START_TIME=$(date +%s)
LAST_EMAIL=0
MAX_RUNTIME=$((4 * 3600 - 300))  # exit 5 min before walltime

# Send initial report on startup
if [[ -f "$REPORT_SCRIPT" ]]; then
    send_report 2>/dev/null || true
    LAST_EMAIL=$(date +%s)
fi

while true; do
    CYCLE=$((CYCLE + 1))
    NOW=$(date '+%H:%M:%S')
    ELAPSED=$(( $(date +%s) - START_TIME ))

    # Time check — resubmit before walltime
    if [[ $ELAPSED -ge $MAX_RUNTIME ]]; then
        echo ""
        echo "[$NOW] Approaching walltime — exiting loop"
        break
    fi

    # ── Gather metrics ────────────────────────────────────────────────
    N_DL=$(count_jobs "scgeo-dl")
    N_QT=$(count_jobs "scgeo-qt")
    N_XL=$(count_jobs "scgeo-qt-xl")
    DISK_GB=$(scratch_gb)
    READY=$(count_ready)
    N_NEEDS_XL=$(count_needs_xl)
    DONE_DL=$(count_success_dl)
    DONE_QT=$(count_quant_done)

    echo "────────────────────────────────────────────────────"
    printf "[%s] Cycle %d | DL:%d QT:%d XL:%d | Disk:%sGB | Ready:%d NeedsXL:%d | Done DL:%d QT:%d\n" \
        "$NOW" "$CYCLE" "$N_DL" "$N_QT" "$N_XL" "$DISK_GB" "$READY" "$N_NEEDS_XL" "$DONE_DL" "$DONE_QT"

    # ── Scratch critical — pause everything ───────────────────────────
    if [[ "$DISK_GB" -ge "$SCRATCH_CRIT_GB" ]]; then
        echo "  ⛔ Scratch CRITICAL (${DISK_GB}GB >= ${SCRATCH_CRIT_GB}GB) — pausing all"
        sleep "$POLL_INTERVAL"
        continue
    fi

    # ── Quant: always try to process ready samples ────────────────────
    if [[ "$READY" -gt 0 ]]; then
        # How many quant jobs do we need?
        NEEDED_QT=$(( (READY + QT_BATCH_SIZE - 1) / QT_BATCH_SIZE ))
        # How many can we add?
        HEADROOM_QT=$(( MAX_QT_JOBS - N_QT ))
        if [[ $HEADROOM_QT -lt 0 ]]; then HEADROOM_QT=0; fi
        TO_SUBMIT_QT=$(( NEEDED_QT < HEADROOM_QT ? NEEDED_QT : HEADROOM_QT ))

        if [[ "$TO_SUBMIT_QT" -gt 0 ]]; then
            echo "  ▸ Launching $TO_SUBMIT_QT quant jobs ($READY ready samples)"
            submit_quants "$TO_SUBMIT_QT"
        fi
    fi

    # ── XL jobs: process OOM-flagged samples ───────────────────────────
    MAX_XL_JOBS="${MAX_XL_JOBS:-10}"
    if [[ "$N_NEEDS_XL" -gt 0 ]]; then
        HEADROOM_XL=$(( MAX_XL_JOBS - N_XL ))
        if [[ $HEADROOM_XL -gt 0 ]]; then
            TO_SUBMIT_XL=$(( N_NEEDS_XL < HEADROOM_XL ? N_NEEDS_XL : HEADROOM_XL ))
            if [[ "$TO_SUBMIT_XL" -gt 0 ]]; then
                echo "  ▸ Launching $TO_SUBMIT_XL XL quant jobs ($N_NEEDS_XL .needs_xl samples)"
                submit_xl_quants "$TO_SUBMIT_XL"
            fi
        fi
    fi

    # ── Downloads: keep the pipeline fed (unless scratch is high) ─────
    if [[ "$DISK_GB" -ge "$SCRATCH_MAX_GB" ]]; then
        echo "  ⚠ Scratch high (${DISK_GB}GB >= ${SCRATCH_MAX_GB}GB) — skipping new downloads"
    else
        # Top up download jobs to MAX_DL_JOBS
        HEADROOM_DL=$(( MAX_DL_JOBS - N_DL ))
        if [[ $HEADROOM_DL -lt 0 ]]; then HEADROOM_DL=0; fi

        # Submit in batches of 25 max per cycle to avoid queue flooding
        if [[ "$HEADROOM_DL" -gt 25 ]]; then HEADROOM_DL=25; fi

        if [[ "$HEADROOM_DL" -gt 0 ]]; then
            echo "  ▸ Topping up $HEADROOM_DL download jobs (current: $N_DL/$MAX_DL_JOBS)"
            submit_downloads "$HEADROOM_DL"
        fi
    fi

    # ── Check if pipeline is drained ──────────────────────────────────
    if [[ "$N_DL" -eq 0 && "$N_QT" -eq 0 && "$READY" -eq 0 ]]; then
        # No jobs, no ready — try to submit one dl job as a probe
        echo "  ℹ Pipeline idle — probing for eligible samples..."
        PROBE_JOB=$(sbatch \
            --export=ALL,PHASE="$PHASE",BATCH_SIZE="$DL_BATCH_SIZE" \
            --parsable \
            "$DL_SCRIPT" 2>&1) || true
        if [[ "$PROBE_JOB" =~ ^[0-9]+$ ]]; then
            echo "  ↻ Probe job $PROBE_JOB submitted — pipeline still has work"
        else
            echo "  ✓ Cannot submit new work — orchestrator complete!"
            RESUBMIT="false"
            break
        fi
    fi
    # ── Hourly email report ────────────────────────────────────────
    NOW_EPOCH=$(date +%s)
    if [[ -f "$REPORT_SCRIPT" ]] && [[ $((NOW_EPOCH - LAST_EMAIL)) -ge $EMAIL_INTERVAL ]]; then
        send_report 2>/dev/null || true
        LAST_EMAIL=$NOW_EPOCH
    fi
    # ── Sleep until next poll ─────────────────────────────────────────
    sleep "$POLL_INTERVAL"
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Orchestrator exiting after $CYCLE cycles"
echo "  Downloads done: $(count_success_dl)"
echo "  Quant done:     $(count_quant_done)"
echo "  .1pz files:     $(find $PROJECT/scgeo/pipeline/quant -name 'counts.1pz' 2>/dev/null | wc -l)"
echo "  Scratch:        $(scratch_gb)GB"
echo "════════════════════════════════════════════════════════════════"

# ── Final report before exit ──────────────────────────────────────────
if [[ -f "$REPORT_SCRIPT" ]]; then
    send_report 2>/dev/null || true
fi

# ── Self-resubmit ─────────────────────────────────────────────────────
if [[ "$RESUBMIT" == "true" ]]; then
    NEXT_JOB=$(sbatch --parsable "$ORCH_SCRIPT" 2>&1) || true
    if [[ "$NEXT_JOB" =~ ^[0-9]+$ ]]; then
        echo "  ↻ Resubmitted as job $NEXT_JOB"
    else
        echo "  ✗ Failed to resubmit: $NEXT_JOB"
    fi
fi
