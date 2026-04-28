#!/bin/bash
# genome_unload.sh — Genome unload helper for SLURM job epilogue
#
# Usage: genome_unload.sh <genome_dir>
#   Call after all jobs on a node are complete to free shared memory.
#   Safe to call even if genome is not currently loaded (no-op).
#
# Example SLURM epilogue usage:
#   bash /mnt/home/debruinz/Singlet-AI/singlify/scripts/genome_unload.sh "$GENOME_DIR"

GENOME_DIR="${1:?Usage: genome_unload.sh <genome_dir>}"
SINGLIFY="${SINGLIFY:-/mnt/home/debruinz/Singlet-AI/singlify/build/singlify}"

if "$SINGLIFY" genome status "$GENOME_DIR" 2>/dev/null | grep -q "loaded"; then
    echo "[genome_unload] Unloading genome from shared memory: $GENOME_DIR"
    "$SINGLIFY" genome unload "$GENOME_DIR"
    echo "[genome_unload] Done"
else
    echo "[genome_unload] Genome not in shared memory, nothing to unload"
fi
