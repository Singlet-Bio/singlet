#!/bin/bash
#
# Unified Pipeline Launch Script
#
# Submits SLURM arrays for any processing phase. The --phase flag selects
# the eligibility filter and all three partition arrays share one worker script.
#
# Usage:
#   bash launch.sh --phase 1                     # Full launch (reconcile + submit)
#   bash launch.sh --phase 1 --submit-only       # Skip reconcile, just submit
#   bash launch.sh --phase 2a --submit-only      # Submit multiome phase
#   bash launch.sh --phase 4a --submit-only      # Submit mouse phase
#
# Array sizing (tuned to FILL ALL NODES at 8 CPUs, 128G per task):
#
# Capacity at 128G/task, 8 CPUs/task (OverSubscribe=NO):
#   cpu:    11 nodes → 58 tasks  (c001-c006: 5/node=30, c007-c010: 6/node=24, c100: 4)
#   bigmem: 3 nodes (QOS limit) → 18 tasks  (6/node, CPU-bound)
#   short:  8 GPU nodes → 33 tasks  (g001-g004: 3/node=12, g005: 3, g050-g052: 6/node=18)
#   Total capacity: 109 tasks on 22 nodes
#
# Throttled to 109 concurrent (fills every available node):
#   cpu:    0-999%58   (58 concurrent, 8 CPUs, 128G)  — all 11 cpu nodes
#   bigmem: 0-199%18   (18 concurrent, 8 CPUs, 128G)  — all 3 bigmem nodes
#   short:  0-999%33   (33 concurrent, 8 CPUs, 128G)  — all 8 GPU nodes
#
# At 109 concurrent: ~65 downloading → ~260 curl connections
# Prior at 72/64G: only used 11 of 22 nodes
# CPUs per task: 8 (simpleaf at 8 threads; substantial speedup over 4)
#
# System limits: MaxArraySize=1001, MaxJobCount=10000, no per-user job limits
# "JobArrayTaskLimit" is just our %N throttle, not a system constraint
#
# All three arrays share the same claim ledger, so tasks never double-claim.
# When a phase has fewer samples than total capacity, tasks exit cleanly when
# no unclaimed samples remain.

set -euo pipefail

SCRIPTS_DIR="/mnt/home/debruinz/Singlet-AI/scripts"
PIPELINE_DIR="/mnt/projects/debruinz_project/cellarium/pipeline"
LOG_DIR="$PIPELINE_DIR/logs"
WORKER="$SCRIPTS_DIR/slurm_worker.sh"

# ── Parse arguments ──
PHASE=""
SUBMIT_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)    PHASE="$2"; shift 2 ;;
        --submit-only) SUBMIT_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: bash launch.sh --phase PHASE [--submit-only]"
            echo "Phases: 1, 2a, 2b, 2c, 2d, 3, 4a, 4b"
            exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$PHASE" ]]; then
    echo "ERROR: --phase is required"
    echo "Usage: bash launch.sh --phase PHASE [--submit-only]"
    exit 1
fi

# Phase names for display
declare -A PHASE_NAMES=(
    [1]="Human droplet RNA"
    [2a]="Human multiome GEX"
    [2b]="Human CITE-seq GEX"
    [2c]="Human reclassified droplet"
    [2d]="Human ambiguous (auto-detect)"
    [3]="Human screen-flagged recovery"
    [4a]="Mouse droplet RNA"
    [4b]="Other model organisms"
)

PHASE_NAME="${PHASE_NAMES[$PHASE]:-Phase $PHASE}"

echo "════════════════════════════════════════════════════"
echo "  Singlet Bio — Pipeline Launch"
echo "  Phase $PHASE: $PHASE_NAME"
echo "  $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════"

# ── Step 1: Reconcile catalog (unless --submit-only) ──
if [[ "$SUBMIT_ONLY" == false ]]; then
    echo ""
    echo "▸ Step 1: Reconciling catalog with on-disk manifests..."
    srun --partition=cpu --cpus-per-task=4 --mem=16G --time=00:30:00 \
        bash -c "
            module load miniconda3/25.5.1
            source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
            conda activate cellarium
            cd /tmp
            python3 $SCRIPTS_DIR/reconcile_catalog.py --apply
        "
    echo "  Reconciliation complete"
fi

# ── Step 2: Create directories ──
echo ""
echo "▸ Setting up pipeline directories..."
mkdir -p "$PIPELINE_DIR/claims"
mkdir -p "$PIPELINE_DIR/results"
mkdir -p "$LOG_DIR"

# ── Step 3: Submit arrays ──
# MaxArraySize=1001 (indices 0-1000), MaxJobCount=10,000
# All three arrays share the same claim ledger — no double-claiming
echo ""
echo "▸ Submitting SLURM arrays for phase $PHASE..."

# 8 CPUs per task — fills all 22 nodes at 128G each.
# 109 concurrent total, ~65 downloading at any time.
COMMON_ARGS="--export=ALL,PHASE=$PHASE --cpus-per-task=8"

CPU_JOB=$(sbatch $COMMON_ARGS \
    --partition=cpu --mem=128G --time=2-00:00:00 \
    --array=0-999%58 \
    --job-name="p${PHASE}-cpu" \
    "$WORKER" 2>&1 | grep -oP '\d+')
echo "  cpu:    job $CPU_JOB  (1000 tasks, %58 concurrent, 8 CPUs, 128G)"

BIG_JOB=$(sbatch $COMMON_ARGS \
    --partition=bigmem --mem=128G --time=2-00:00:00 \
    --array=0-199%18 \
    --job-name="p${PHASE}-big" \
    "$WORKER" 2>&1 | grep -oP '\d+')
echo "  bigmem: job $BIG_JOB  (200 tasks, %18 concurrent, 8 CPUs, 128G)"

SHORT_JOB=$(sbatch $COMMON_ARGS \
    --partition=short --mem=128G --time=1-00:00:00 \
    --array=0-999%33 \
    --job-name="p${PHASE}-short" \
    "$WORKER" 2>&1 | grep -oP '\d+')
echo "  short:  job $SHORT_JOB  (1000 tasks, %33 concurrent, 8 CPUs, 128G)"

# ── Step 4: Start monitor ──
echo ""
echo "▸ Starting monitor..."
MON_JOB=$(sbatch --begin=now+30minutes \
    --export=ALL,PHASE=$PHASE \
    "$SCRIPTS_DIR/slurm_monitor.sh" 2>&1 | grep -oP '\d+')
echo "  monitor: job $MON_JOB  (first email in 30 min, then every 3h)"

# ── Summary ──
echo ""
echo "════════════════════════════════════════════════════"
echo "  Phase $PHASE Launched: $PHASE_NAME"
echo ""
echo "  Jobs:"
echo "    cpu:     $CPU_JOB  (1000 × 5 = 5,000 capacity)"
echo "    bigmem:  $BIG_JOB  (200 × 5 = 1,000 capacity)"
echo "    short:   $SHORT_JOB  (1000 × 5 = 5,000 capacity)"
echo "    monitor: $MON_JOB"
echo ""
echo "  Max concurrency: 58 + 18 + 33 = 109 tasks (8 CPUs, 128G, all 22 nodes)"
echo ""
echo "  Monitor: squeue -u $USER"
echo "  Status:  python3 $SCRIPTS_DIR/grab_batch.py --status"
echo "  Report:  python3 $SCRIPTS_DIR/pipeline_monitor.py"
echo "════════════════════════════════════════════════════"
