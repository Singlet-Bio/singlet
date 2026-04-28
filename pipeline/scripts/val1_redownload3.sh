#!/bin/bash
#SBATCH --job-name=val1-redl3
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/redl3_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/redl3_%a.log
#SBATCH --array=52,56,116
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

OUTDIR="${OUTBASE}/${SRR}"
mkdir -p "$OUTDIR"
OUTFILE="${OUTDIR}/${SRR}.1fq"

rm -f "$OUTFILE" 2>/dev/null || true
rm -rf "${OUTDIR}/output" 2>/dev/null || true

echo "START: $SRR protocol=$PROTOCOL idx=$SLURM_ARRAY_TASK_ID $(date)"

"$SINGLIFY" download "$SRR" \
    --output "$OUTFILE" \
    --protocol "$PROTOCOL" \
    --vdb-threads 4 \
    2>&1

echo "DONE: $SRR size=$(stat -c%s "$OUTFILE" 2>/dev/null || echo 0) $(date)"
