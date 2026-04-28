#!/bin/bash
#SBATCH --job-name=val1-redl4
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/redl4_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/redl4_%a.log
#SBATCH --array=3,4,5,9,10,14,15,21,23,24,26,28,36,38,39,49,52,56,63,71,75,77,82,83,107,111,115,116,119,121,122,123,124,127,130,135,136,137,152,153,157,158,159,161,164%5
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=cpu

# Re-download all 45 human+mouse samples that failed processing
# Previous batch used 32G which was insufficient for large samples
# Bumped to 128G to handle samples up to ~300M reads

set -euo pipefail

export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}

SAMPLE_CSV="/mnt/home/debruinz/Singlet-AI/singlify/scripts/val1_samples.csv"
SINGLIFY="/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
OUTBASE="/mnt/projects/debruinz_project/singlify_validation/val1"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SAMPLE_CSV")
SRR=$(echo "$LINE" | cut -d, -f1)

if [ -z "$SRR" ] || [ "$SRR" = "srr_accession" ]; then
    echo "SKIP: header or empty line $SLURM_ARRAY_TASK_ID"
    exit 0
fi

SRRDIR="${OUTBASE}/${SRR}"
OUTFQ="${SRRDIR}/${SRR}.1fq"

# Delete old corrupt .1fq
rm -f "$OUTFQ"

echo "[$(date -Iseconds)] START: re-download $SRR idx=$SLURM_ARRAY_TASK_ID"

mkdir -p "$SRRDIR"
/usr/bin/time -f "wall=%e rss=%MKB" \
  $SINGLIFY download "$SRR" --output "$OUTFQ" --vdb-threads 4 2>&1

if [ -f "$OUTFQ" ] && [ -s "$OUTFQ" ]; then
    echo "[$(date -Iseconds)] DONE: $SRR $(du -sh "$OUTFQ" | awk '{print $1}')"
else
    echo "[$(date -Iseconds)] FAIL: $SRR — no output file"
    exit 1
fi
