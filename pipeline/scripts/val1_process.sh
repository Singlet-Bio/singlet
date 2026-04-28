#!/bin/bash
#SBATCH --job-name=val1-proc
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/proc_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/proc_%a.log
#SBATCH --array=2-178%20
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=384G
#SBATCH --partition=cpu

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

DATADIR="${OUTBASE}/${SRR}"
OUTDIR="${DATADIR}/output"
INFILE="${DATADIR}/${SRR}.1fq"

if [ ! -f "$INFILE" ]; then
    echo "ERROR: $SRR .1fq not found at $INFILE"
    exit 1
fi

# Skip if output already complete
if [ -f "${OUTDIR}/pileup_stats.json" ]; then
    echo "SKIP: $SRR output already exists"
    exit 0
fi

# Select genome and GTF based on species (only human and mouse currently indexed)
case "$SPECIES" in
    "Homo sapiens")
        GENOME="${REFBASE}/GRCh38-2024-A/star_2.7.11b"
        GTF="${REFBASE}/GRCh38-2024-A/genes/genes.gtf"
        ;;
    "Mus musculus")
        GENOME="${REFBASE}/GRCm39-2024-A/star_2.7.11b"
        GTF="${REFBASE}/GRCm39-2024-A/genes/genes.gtf"
        ;;
    *)
        echo "SKIP: $SRR unsupported species '$SPECIES' (no STAR index available)"
        exit 0
        ;;
esac

mkdir -p "$OUTDIR"

echo "START: $SRR species=$SPECIES $(date)"

# --genome-shared: reuse genome pre-loaded into POSIX SHM via
#   singlify genome load --genome-dir $GENOME
# N21 auto-detects SHM even without this flag, but explicit is safer.
/usr/bin/time -f "wall=%e rss=%MKB" \
    "$SINGLIFY" "$INFILE" \
    --genome-dir "$GENOME" \
    --exons "$GTF" \
    --whitelist None \
    --out-prefix "${OUTDIR}/" \
    --threads 20 \
    --genome-shared \
    2>&1

EXIT=$?
echo "DONE: $SRR exit=${EXIT} $(date)"
exit $EXIT
