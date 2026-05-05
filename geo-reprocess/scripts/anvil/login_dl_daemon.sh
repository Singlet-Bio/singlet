#!/usr/bin/env bash
# ── Login-node download daemon ───────────────────────────────────────
# Maintains N concurrent single-threaded downloads on a login node.
# When a download finishes, submits a quant SLURM job and starts the
# next download — keeping steady concurrency until no eligible samples
# remain.
#
# Usage (inside tmux on a login node):
#   bash login_dl_daemon.sh              # 20 workers (default)
#   WORKERS=10 bash login_dl_daemon.sh   # 10 workers
#
# Zero SU cost for downloads. Quant jobs go to SLURM shared partition.
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
export SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"
WORKERS="${WORKERS:-20}"
STAGGER="${STAGGER:-8}"         # seconds between initial launches
POLL="${POLL:-5}"               # seconds between checking for finished workers
MAX_IDLE="${MAX_IDLE:-3600}"    # P3: exit after this many seconds with no work (1 hr)
SUBMIT_QUANT="${SUBMIT_QUANT:-true}"  # submit quant jobs after each DL
MAX_QUANT_XL="${MAX_QUANT_XL:-2}"     # max concurrent scgeo-qt-xl jobs (OOM retry)

NODE=$(hostname | cut -d. -f1)
SCRIPT_DIR="$PROJECT/geo-reprocess/scripts/anvil"
DL_SCRIPT="$SCRIPT_DIR/login_download.sh"
QT_SCRIPT="$SCRIPT_DIR/anvil_quant.sh"
SRA_QT_SCRIPT="$SCRIPT_DIR/anvil_quant_sra.sh"
XL_QT_SCRIPT="$SCRIPT_DIR/anvil_quant_xl.sh"
LOG_DIR="$PROJECT/scgeo/pipeline/logs/login_dl"
DL_DIR="$SCRATCH/scgeo_downloads"

mkdir -p "$LOG_DIR"

DAEMON_LOG="$LOG_DIR/daemon_${NODE}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$DAEMON_LOG"; }

log "════════════════════════════════════════════════════"
log "Login DL daemon | $NODE | PID $$ | workers=$WORKERS"
log "════════════════════════════════════════════════════"

# Load environment once (children inherit)
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    module load anaconda 2>/dev/null || true
    conda activate "$PROJECT/envs/scgeo"
fi
umask 0022  # Reset after conda activate (may set 0117)

# ── Worker management ─────────────────────────────────────────────────
# Arrays: PIDS[slot]=pid, LOGS[slot]=logfile, STARTS[slot]=epoch
declare -A PIDS LOGS STARTS
SLOT_SEQ=0        # monotonic slot counter for unique log names
TOTAL_DONE=0
TOTAL_FAIL=0
IDLE_SINCE=""

# Launch one download worker in a slot
launch_worker() {
    local slot="$1"
    SLOT_SEQ=$((SLOT_SEQ + 1))
    local logfile="$LOG_DIR/${NODE}_d${SLOT_SEQ}.log"

    BATCH_SIZE=1 bash "$DL_SCRIPT" > "$logfile" 2>&1 &
    local pid=$!
    PIDS[$slot]=$pid
    LOGS[$slot]=$logfile
    STARTS[$slot]=$(date +%s)
    log "  slot $slot → PID $pid (log: $(basename "$logfile"))"
}

# Check if worker in slot finished; handle result
check_slot() {
    local slot="$1"
    local pid="${PIDS[$slot]}"
    if ! kill -0 "$pid" 2>/dev/null; then
        # Process finished
        wait "$pid" 2>/dev/null || true
        local logfile="${LOGS[$slot]}"
        local started="${STARTS[$slot]}"
        local elapsed=$(( $(date +%s) - started ))

        # Determine outcome from log
        if grep -q 'Login download complete' "$logfile" 2>/dev/null; then
            local dl_ok=$(grep -oP 'Downloaded: \K\d+' "$logfile" 2>/dev/null || echo 0)
            local dl_fail=$(grep -oP 'Failed: \K\d+' "$logfile" 2>/dev/null || echo 0)
            TOTAL_DONE=$((TOTAL_DONE + dl_ok))
            TOTAL_FAIL=$((TOTAL_FAIL + dl_fail))
            log "  slot $slot done (${elapsed}s) ok=$dl_ok fail=$dl_fail [total: $TOTAL_DONE done, $TOTAL_FAIL fail]"

            # Submit quant job if download succeeded (no cap — SLURM handles backpressure)
            if [[ "$SUBMIT_QUANT" == "true" && "$dl_ok" -gt 0 ]]; then
                # P4: check if the new download is an SRA-pending sample
                # (login_download.sh writes method=s3_sra_pending to manifest)
                local sra_pending
                sra_pending=$(find "$DL_DIR" -name "download_manifest.json" -newer "$logfile" \
                    -exec grep -l '"method": "s3_sra_pending"' {} \; 2>/dev/null | wc -l)

                if [[ "$sra_pending" -gt 0 ]]; then
                    # Route to dedicated SRA quant job (QUANT_BATCH=1, 24h, 32G)
                    local sra_id
                    sra_id=$(sbatch --export=ALL,QUANT_BATCH=1,FASTQ_ONLY=false \
                        --parsable "$SRA_QT_SCRIPT" 2>&1) || true
                    if [[ "$sra_id" =~ ^[0-9]+$ ]]; then
                        log "    → SRA quant job $sra_id submitted (fasterq-dump path)"
                    else
                        log "    → SRA quant submit failed: $sra_id"
                    fi
                else
                    # FASTQ-ready sample: fast batch quant job
                    local qt_id
                    qt_id=$(sbatch --export=ALL,QUANT_BATCH=5,FASTQ_ONLY=true \
                        --parsable "$QT_SCRIPT" 2>&1) || true
                    if [[ "$qt_id" =~ ^[0-9]+$ ]]; then
                        log "    → quant job $qt_id submitted"
                    else
                        log "    → quant submit failed: $qt_id"
                    fi
                fi

                # Submit XL job if OOM-marked samples exist and below XL cap
                local xl_count xl_running
                xl_count=$(find "$DL_DIR" -name ".needs_xl" 2>/dev/null | wc -l)
                if [[ "$xl_count" -gt 0 && -f "$XL_QT_SCRIPT" ]]; then
                    xl_running=$(squeue -u "$(whoami)" -n scgeo-qt-xl -h 2>/dev/null | wc -l)
                    if [[ "$xl_running" -lt "$MAX_QUANT_XL" ]]; then
                        local xl_id
                        xl_id=$(sbatch --parsable "$XL_QT_SCRIPT" 2>&1) || true
                        [[ "$xl_id" =~ ^[0-9]+$ ]] && log "    → XL quant job $xl_id submitted ($xl_count xl samples pending)"
                    fi
                fi
            fi
        elif grep -q 'No unclaimed eligible samples' "$logfile" 2>/dev/null; then
            log "  slot $slot: no eligible samples (${elapsed}s)"
            unset "PIDS[$slot]"
            return 1  # signal: no work left
        else
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            local err=$(tail -1 "$logfile" 2>/dev/null)
            log "  slot $slot error (${elapsed}s): $err [total: $TOTAL_DONE done, $TOTAL_FAIL fail]"
        fi

        unset "PIDS[$slot]"
        return 0  # signal: slot free, work may remain
    fi
    return 2  # signal: still running
}

# ── Initial launch with stagger ───────────────────────────────────────
log "Launching $WORKERS workers (stagger=${STAGGER}s)..."
for slot in $(seq 1 "$WORKERS"); do
    launch_worker "$slot"
    if [[ "$slot" -lt "$WORKERS" ]]; then
        sleep "$STAGGER"
    fi
done
log "All $WORKERS initial workers launched"

# ── Main loop: monitor and refill ─────────────────────────────────────
NO_WORK_COUNT=0

while true; do
    sleep "$POLL"

    ACTIVE=0
    FREED=0
    NO_WORK_THIS_CYCLE=0

    for slot in "${!PIDS[@]}"; do
        check_slot "$slot" && rc=$? || rc=$?
        if [[ $rc -eq 0 ]]; then
            # Slot freed, work may remain — refill
            FREED=$((FREED + 1))
            launch_worker "$slot"
            sleep 1  # tiny stagger between refills
        elif [[ $rc -eq 1 ]]; then
            # No eligible samples left
            NO_WORK_THIS_CYCLE=$((NO_WORK_THIS_CYCLE + 1))
        else
            # Still running
            ACTIVE=$((ACTIVE + 1))
        fi
    done

    # If some slots freed with no work, don't refill those
    ACTIVE=${#PIDS[@]}

    if [[ $ACTIVE -eq 0 ]]; then
        # All workers finished and none could get new work
        if [[ -z "$IDLE_SINCE" ]]; then
            IDLE_SINCE=$(date +%s)
            log "All workers idle — waiting up to ${MAX_IDLE}s for new eligible samples..."
        fi
        IDLE_TIME=$(( $(date +%s) - IDLE_SINCE ))
        if [[ $IDLE_TIME -ge $MAX_IDLE ]]; then
            log "Idle timeout (${MAX_IDLE}s) — shutting down daemon"
            break
        fi
        # Probe: try launching one worker to check if samples became available
        if [[ $((IDLE_TIME % 60)) -lt $POLL ]]; then
            launch_worker "probe"
        fi
    else
        IDLE_SINCE=""
    fi

    log "STATUS: active=$ACTIVE done=$TOTAL_DONE fail=$TOTAL_FAIL slots_refilled=$FREED"
done

# ── Cleanup: wait for any remaining workers ───────────────────────────
log "Waiting for remaining workers..."
for slot in "${!PIDS[@]}"; do
    wait "${PIDS[$slot]}" 2>/dev/null || true
done

log "════════════════════════════════════════════════════"
log "Daemon shutdown | $NODE | done=$TOTAL_DONE fail=$TOTAL_FAIL"
log "════════════════════════════════════════════════════"
