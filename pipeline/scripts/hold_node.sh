#!/bin/bash
# hold_node.sh — submit a 72-hour SLURM job to keep a compute node available
# for the singlify-perf autonomous run.
#
# Usage:
#   bash hold_node.sh [--hours 72] [--node c006]
#
# The job does nothing (sleep) but holds the node allocation so SSH access
# is available for builds and benchmarks throughout the 3-day sprint.
# The job writes a heartbeat line every 30 minutes so SLURM knows it's alive.
#
# Check status: squeue -u debruinz | grep singlify_hold

HOURS=72
NODE_CONSTRAINT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours) HOURS="$2"; shift 2 ;;
        --node)  NODE_CONSTRAINT="#SBATCH --nodelist=$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

SCRIPT=$(mktemp /tmp/hold_node_XXXXXX.sh)
cat > "$SCRIPT" << SLURM
#!/bin/bash
#SBATCH --job-name=singlify_hold
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=${HOURS}:00:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/scripts/hold_node_%j.log
${NODE_CONSTRAINT}

echo "singlify-perf hold node started: \$(hostname) at \$(date)"
echo "Job will hold until: \$(date -d '+${HOURS} hours')"

# Heartbeat loop — keeps job alive, records timestamps
SECONDS_TO_HOLD=\$(( ${HOURS} * 3600 ))
ELAPSED=0
while [[ \$ELAPSED -lt \$SECONDS_TO_HOLD ]]; do
    sleep 1800   # 30 minutes
    ELAPSED=\$(( ELAPSED + 1800 ))
    echo "[\$(date '+%Y-%m-%d %H:%M')] heartbeat on \$(hostname), elapsed=${math}\${ELAPSED}s"
done
echo "singlify-perf hold node job ended normally."
SLURM

# Submit
JOB_ID=$(sbatch "$SCRIPT" | awk '{print $NF}')
rm -f "$SCRIPT"

if [[ -z "$JOB_ID" ]]; then
    echo "ERROR: sbatch failed"
    exit 1
fi

echo "Submitted hold node job: JOBID=$JOB_ID"
echo "Check: squeue -u debruinz -j $JOB_ID"
echo ""
echo "Once RUNNING, find the node with:"
echo "  squeue -u debruinz -j $JOB_ID --format='%N'"
echo ""
echo "Then SSH to it:"
echo "  ssh <node> 'hostname'"
