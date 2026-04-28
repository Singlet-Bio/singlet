#!/bin/bash
#SBATCH --job-name=val1-reproc
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/reproc_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/reproc_%a.log
#SBATCH --array=3,4,10,15,16,21,24,26,28,36,38,39,44,49,55,63,64,65,68,70,71,75,76,77,79,80,81,84,85,86,102,107,119,121,122,123,124,127,130,136,137,148,152,153,155,159,161,162,164%3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=384G
#SBATCH --partition=cpu
#SBATCH --dependency=afterany:350913:350920

set -euo pipefail

source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SAMPLE_CSV="/mnt/home/debruinz/Singlet-AI/singlify/scripts/val1_samples.csv"
SINGLIFY="/mnt/home/debruinz/Singlet-AI/singlify/build/singlify"
OUTBASE="/mnt/projects/debruinz_project/singlify_validation/val1"
REFBASE="/mnt/projects/debruinz_project/cellarium/reference"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SAMPLE_CSV")
SRR=$(echo "$LINE" | cut -d, -f1)
SPECIES=$(echo "$LINE" | cut -d, -f4)

if [ -z "$SRR" ] || [ "$SRR" = "srr_accession" ]; then
    echo "SKIP: header or empty line $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Skip if already processed
if [ -f "${OUTBASE}/${SRR}/output/pileup_stats.json" ]; then
    echo "SKIP: $SRR already processed (pileup_stats.json exists)"
    exit 0
fi

# Check .1fq exists
FQ="${OUTBASE}/${SRR}/${SRR}.1fq"
if [ ! -f "$FQ" ]; then
    echo "ERROR: .1fq missing for $SRR (download may have failed)"
    exit 1
fi

# Select reference
case "$SPECIES" in
    "Homo sapiens")
        REFDIR="${REFBASE}/GRCh38-2024-A"
        ;;
    "Mus musculus")
        REFDIR="${REFBASE}/GRCm39-2024-A"
        ;;
    *)
        echo "SKIP: $SRR unsupported species '$SPECIES' (no STAR index available)"
        exit 0
        ;;
esac

OUTDIR="${OUTBASE}/${SRR}/output"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "START: $SRR species=$SPECIES idx=$SLURM_ARRAY_TASK_ID $(date)"

/usr/bin/time -f "wall=%e rss=%MKB" "$SINGLIFY" "$FQ" \
    --genome-dir "${REFDIR}/star_2.7.11b" \
    --exons "${REFDIR}/genes/genes.gtf" \
    --whitelist None \
    --out-prefix "${OUTDIR}/" \
    --threads 20 \
    2>&1

echo "DONE: $SRR $(date)"
