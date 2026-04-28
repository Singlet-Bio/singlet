#!/bin/bash
#SBATCH --job-name=val1-redl
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/redl_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/redl_%a.log
#SBATCH --array=3,10,15,21,26,28,36,39,49,55,63,68,71,75,77,119,121,122,123,124,127,130,136,137,152,153,155,159,161,164%5
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu

set -euo pipefail

export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH

SAMPLE_CSV="/mnt/home/debruinz/Singlet-AI/singlify/scripts/val1_samples.csv"
SINGLIFY="/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
OUTBASE="/mnt/projects/debruinz_project/singlify_validation/val1"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SAMPLE_CSV")
SRR=$(echo "$LINE" | cut -d, -f1)
PROTOCOL=$(echo "$LINE" | cut -d, -f3)

if [ -z "$SRR" ] || [ "$SRR" = "srr_accession" ]; then
    echo "SKIP: header or empty line $SLURM_ARRAY_TASK_ID"
    exit 0
fi

OUTDIR="${OUTBASE}/${SRR}"
mkdir -p "$OUTDIR"
OUTFILE="${OUTDIR}/${SRR}.1fq"

# Force re-download: delete existing (potentially corrupt) file
if [ -f "$OUTFILE" ]; then
    echo "DELETE corrupt: $OUTFILE ($(stat -c%s "$OUTFILE") bytes)"
    rm -f "$OUTFILE"
fi

# Also clean any stale output
rm -rf "${OUTDIR}/output" 2>/dev/null || true

echo "START: $SRR protocol=$PROTOCOL idx=$SLURM_ARRAY_TASK_ID $(date)"

"$SINGLIFY" download "$SRR" \
    --output "$OUTFILE" \
    --protocol "$PROTOCOL" \
    --vdb-threads 4 \
    2>&1

SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || echo 0)
echo "DONE: $SRR size=${SIZE} $(date)"
