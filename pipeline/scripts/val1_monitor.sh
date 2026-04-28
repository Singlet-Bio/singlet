#!/bin/bash
# VAL1 Monitor — auto-submits process jobs for newly-downloaded samples
# Run once, checks all logs, submits any unprocessed samples
# Usage: bash scripts/val1_monitor.sh

cd "$(dirname "$0")/.."
LOGDIR="val1_logs"
SCRIPT="scripts/val1_process.sh"

READY=""
for n in $(seq 2 178); do
    DL_LOG="$LOGDIR/dl_$n.log"
    PROC_LOG="$LOGDIR/proc_$n.log"
    
    # Check: download done AND (no proc log OR proc log has error/OOM but not DONE/SKIP)
    if [ -f "$DL_LOG" ] && grep -q "DONE:" "$DL_LOG" 2>/dev/null; then
        if [ ! -f "$PROC_LOG" ]; then
            READY="$READY,$n"
        elif ! grep -q "DONE:\|SKIP:" "$PROC_LOG" 2>/dev/null; then
            # Has proc log but no success — check if it's OOM or error
            if grep -q "oom_kill\|bad_alloc\|error\|signal\|malloc\|SIGSEGV\|SIGABRT" "$PROC_LOG" 2>/dev/null; then
                READY="$READY,$n"
            fi
        fi
    fi
done

READY="${READY#,}"  # Remove leading comma

if [ -z "$READY" ]; then
    echo "No new samples to process"
else
    COUNT=$(echo "$READY" | tr ',' '\n' | wc -l)
    echo "Submitting $COUNT samples: $READY"
    sbatch --array=$READY%20 "$SCRIPT"
fi

# Also print current status
echo ""
echo "=== STATUS ==="
DL_DONE=$(grep -rl "DONE:" $LOGDIR/dl_*.log 2>/dev/null | wc -l)
PROC_DONE=$(grep -rl "DONE:" $LOGDIR/proc_*.log 2>/dev/null | wc -l)
PROC_SKIP=$(grep -rl "SKIP:" $LOGDIR/proc_*.log 2>/dev/null | wc -l)
PROC_OOM=$(grep -rl "oom_kill\|bad_alloc" $LOGDIR/proc_*.log 2>/dev/null | wc -l)
echo "Downloads:  $DL_DONE / 177 complete"
echo "Processed:  $PROC_DONE success, $PROC_SKIP skipped, $PROC_OOM OOM"
echo "Active:     $(squeue -u $(whoami) 2>/dev/null | grep val1 | wc -l) SLURM jobs"
