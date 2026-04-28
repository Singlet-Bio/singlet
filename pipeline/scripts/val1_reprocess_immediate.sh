#!/bin/bash
#SBATCH --job-name=val1-riproc
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/riproc_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/riproc_%a.log
#SBATCH --array=3,4,5,9,10,14,15,21,23,24,26,28,36,38,39,49,52,56,71,77,82%5
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=384G
#SBATCH --partition=cpu

# Immediate reprocess for 21 freshly re-downloaded samples (from batch 352406)
# No dependency — these downloads already completed successfully

set -euo pipefail

export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}

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

SDIR="$OUTBASE/$SRR"
FQ="$SDIR/${SRR}.1fq"

if [ ! -f "$FQ" ]; then
    echo "ERROR: .1fq not found: $FQ"
    exit 1
fi

# Skip if already processed successfully
if [ -f "$SDIR/output/pileup_stats.json" ]; then
    echo "SKIP: already processed $SRR"
    exit 0
fi

# Determine genome directory based on species
case "$SPECIES" in
    Homo_sapiens|human)
        GENOME="$REFBASE/GRCh38-2024-A/star_2.7.11b"
        GTF="$REFBASE/GRCh38-2024-A/genes/genes.gtf"
        ;;
    Mus_musculus|mouse)
        GENOME="$REFBASE/GRCm39-2024-A/star_2.7.11b"
        GTF="$REFBASE/GRCm39-2024-A/genes/genes.gtf"
        ;;
    *)
        echo "SKIP: unsupported species $SPECIES for $SRR"
        exit 0
        ;;
esac

mkdir -p "$SDIR/output"

echo "Processing $SRR (species=$SPECIES, .1fq=$(du -sh "$FQ" | cut -f1))"
echo "Start: $(date)"

$SINGLIFY "$FQ" \
    --genome-dir "$GENOME" \
    --exons "$GTF" \
    --out-prefix "$SDIR/output" \
    --threads 20

echo "Done: $(date)"
echo "Exit: $?"
