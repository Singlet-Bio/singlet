#!/bin/bash
#SBATCH --job-name=val1-reproc2
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/reproc2_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/reproc2_%a.log
#SBATCH --array=3,4,5,9,10,14,15,21,23,24,26,28,36,38,39,49,52,56,63,71,75,77,82,83,107,111,115,116,119,121,122,123,124,127,130,135,136,137,152,153,157,158,159,161,164%3
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=384G
#SBATCH --partition=cpu
# Dependency: set after submission with --dependency=afterany:<redl4_jobid>

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
    echo "Already processed: $SRR"
    exit 0
fi

SRRDIR="${OUTBASE}/${SRR}"
OUTFQ="${SRRDIR}/${SRR}.1fq"

if [ ! -f "$OUTFQ" ] || [ ! -s "$OUTFQ" ]; then
    echo "ERROR: .1fq file missing for $SRR"
    exit 1
fi

# Determine genome reference
case "$SPECIES" in
    "Homo sapiens")
        GENOME_DIR="${REFBASE}/GRCh38-2024-A/star_2.7.11b"
        GTF="${REFBASE}/GRCh38-2024-A/genes/genes.gtf"
        ;;
    "Mus musculus")
        GENOME_DIR="${REFBASE}/GRCm39-2024-A/star_2.7.11b"
        GTF="${REFBASE}/GRCm39-2024-A/genes/genes.gtf"
        ;;
    *)
        echo "SKIP: unsupported species '$SPECIES' for $SRR"
        exit 0
        ;;
esac

OUTDIR="${SRRDIR}/output"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "[$(date -Iseconds)] START: $SRR species=$SPECIES idx=$SLURM_ARRAY_TASK_ID"

/usr/bin/time -f "wall=%e rss=%MKB" \
  $SINGLIFY "$OUTFQ" \
    --genome-dir "$GENOME_DIR" \
    --exons "$GTF" \
    --whitelist None \
    --out-prefix "$OUTDIR/" \
    --threads 20 2>&1

if [ -f "$OUTDIR/pileup_stats.json" ]; then
    MAP_PCT=$(grep "Uniquely mapped reads %" "$OUTDIR/star_Log.final.out" 2>/dev/null | awk '{print $NF}' || echo "N/A")
    echo "[$(date -Iseconds)] DONE: $SRR mapping=$MAP_PCT"
else
    echo "[$(date -Iseconds)] FAIL: $SRR — no pileup_stats.json"
    exit 1
fi
