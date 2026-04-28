#!/bin/bash
# genome_preload.sh — Genome preload helper for SLURM job prologue
#
# Usage: source genome_preload.sh <genome_dir>
#   After sourcing, $GENOME_SHARED_FLAG is set to "--genome-shared" if the
#   genome was successfully loaded, or "" if loading failed (fallback to
#   per-job load).
#
# Why: Without --genome-shared, each job loads ~35GB of genome into its own
#   RSS. At 128G total job memory, large samples (>100M reads) OOM because
#   STAR BAM sort (~100GB) + genome (35GB) = 135GB+ peak.
#   With --genome-shared, peak RSS drops from ~100G to ~65G.
#
# Example SLURM job usage:
#   source /mnt/home/debruinz/Singlet-AI/singlify/scripts/genome_preload.sh "$GENOME_DIR"
#   $SINGLIFY "$INPUT_1FQ" \
#       --genome-dir "$GENOME_DIR" \
#       $GENOME_SHARED_FLAG \
#       --exons "$GTF" ...

GENOME_DIR="${1:?Usage: source genome_preload.sh <genome_dir>}"
SINGLIFY="${SINGLIFY:-/mnt/home/debruinz/Singlet-AI/singlify/build/singlify}"

# Check if genome is already loaded in shared memory
if "$SINGLIFY" genome status "$GENOME_DIR" 2>/dev/null | grep -q "loaded"; then
    echo "[genome_preload] Genome already loaded in shared memory: $GENOME_DIR"
    export GENOME_SHARED_FLAG="--genome-shared"
else
    echo "[genome_preload] Loading genome into shared memory: $GENOME_DIR"
    if "$SINGLIFY" genome load "$GENOME_DIR"; then
        echo "[genome_preload] Genome loaded successfully"
        export GENOME_SHARED_FLAG="--genome-shared"
    else
        echo "[genome_preload] WARNING: genome load failed, falling back to per-job loading (no --genome-shared)"
        export GENOME_SHARED_FLAG=""
    fi
fi
