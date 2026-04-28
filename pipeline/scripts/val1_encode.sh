#!/bin/bash
#SBATCH --job-name=val1-enc
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/enc_%a.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlify/val1_logs/enc_%a.log
#SBATCH --array=2-178%10
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu

# Encode pre-downloaded FASTQs → .1fq
# Use this only when FASTQs were obtained outside singlify download
# (e.g. from the GEO pipeline or manual fasterq-dump)

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
OUTFILE="${OUTDIR}/${SRR}.1fq"

# Skip if .1fq already exists
if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
    echo "SKIP: $SRR .1fq already exists"
    exit 0
fi

# Count FASTQ files
NFILES=$(ls "${OUTDIR}/${SRR}"_*.fastq 2>/dev/null | wc -l)
NFILES_GZ=$(ls "${OUTDIR}/${SRR}"_*.fastq.gz 2>/dev/null | wc -l)
[ "$NFILES" -eq 0 ] && NFILES=$NFILES_GZ

echo "START: $SRR protocol=$PROTOCOL nfiles=$NFILES $(date)"

find_reads() {
    local pat
    for pat in "${OUTDIR}/${SRR}_${1}.fastq" "${OUTDIR}/${SRR}_${1}.fastq.gz"; do
        [ -f "$pat" ] && echo "$pat" && return
    done
    echo ""
}

R1=$(find_reads 1)
R2=$(find_reads 2)
R3=$(find_reads 3)

if [ -z "$R1" ] || [ -z "$R2" ]; then
    echo "ERROR: $SRR missing R1/R2 in $OUTDIR"
    exit 1
fi

READS_ARGS="--reads $R1 $R2"
[ -n "$R3" ] && READS_ARGS="--reads $R1 $R2 $R3"

"$SINGLIFY" encode $READS_ARGS \
    --protocol "$PROTOCOL" \
    --output "$OUTFILE" \
    2>&1

SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || echo 0)
echo "DONE: $SRR size=${SIZE} $(date)"
