#!/usr/bin/env bash
# corpus_download.sh — Download one SRA sample + FASTQ for ground truth
# Submitted as individual SLURM jobs
# Usage: sbatch --export=SRR=SRR32855204 corpus_download.sh
set -euo pipefail

CORPUS=/mnt/projects/debruinz_project/singlify_validation/corpus
PROFILES=/mnt/projects/debruinz_project/singlify_validation/profiles
SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
OFQ=/mnt/home/debruinz/Singlet-AI/singlify/build/1fq
FASTERQ=/mnt/home/debruinz/.conda/envs/cellarium/bin/fasterq-dump
export TMPDIR=/dev/shm

echo "[$(date)] Downloading $SRR"

# 1. Download .1fq via singlify
if [[ ! -f "$CORPUS/${SRR}.1fq" ]] || [[ $(stat -c%s "$CORPUS/${SRR}.1fq") -lt 100000 ]]; then
    echo "  Downloading .1fq..."
    /usr/bin/time -v "$SINGLIFY" download "$SRR" -o "$CORPUS/${SRR}.1fq" 2>&1 | tee "$PROFILES/${SRR}_download.log"
else
    echo "  .1fq already exists: $(ls -lh $CORPUS/${SRR}.1fq | awk '{print $5}')"
fi

# 2. Inspect .1fq
echo "  Inspecting .1fq..."
"$OFQ" inspect "$CORPUS/${SRR}.1fq" 2>&1 | tee "$PROFILES/${SRR}_inspect.log"

# 3. Download raw FASTQ (for STARsolo/salmon ground truth)
FASTQ_DIR="$CORPUS/${SRR}_fastq"
mkdir -p "$FASTQ_DIR"

# Count existing FASTQ files
n_existing=$(find "$FASTQ_DIR" -name "*.fastq" -type f 2>/dev/null | wc -l)
if [[ $n_existing -lt 2 ]]; then
    echo "  Downloading FASTQs..."
    rm -f "$FASTQ_DIR"/*.fastq  # Clean partial downloads
    rm -rf "$FASTQ_DIR"/fasterq.tmp.*
    /usr/bin/time -v "$FASTERQ" --threads 4 --split-files --outdir "$FASTQ_DIR" "$SRR" 2>&1 | tee "$PROFILES/${SRR}_fasterq.log"
else
    echo "  FASTQs already exist: $n_existing files"
fi

# 4. Report results
echo ""
echo "=== $SRR Download Summary ==="
echo ".1fq: $(ls -lh $CORPUS/${SRR}.1fq 2>/dev/null | awk '{print $5}')"
echo "FASTQs:"
for f in "$FASTQ_DIR"/*.fastq; do
    len=$(head -2 "$f" | tail -1 | wc -c)
    reads=$(( $(wc -l < "$f") / 4 ))
    echo "  $(basename $f): ${reads} reads, read_len=${len}bp, $(du -h $f | awk '{print $1}')"
done

echo "[$(date)] Done: $SRR"
