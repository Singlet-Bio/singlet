#!/bin/bash
# slurm_genome_prolog.sh — SLURM prolog: load genome if first job on this node
#
# Usage: pass as --prolog in sbatch, or invoke directly in job script.
#   Loads the species genome(s) into /dev/shm shared memory so that
#   subsequent singlify process calls can use --genome-shared and skip
#   the ~40s genome load per sample.
#
# SINGLIFY_REF_BASE must be set (either in environment or below).
#
# Multiple species: add additional genome_load calls for each species
#   expected in the batch (human is always loaded; add mouse for mixed batches).

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
SINGLIFY_REF_BASE="${SINGLIFY_REF_BASE:-/mnt/projects/debruinz_project/cellarium/reference}"

load_genome() {
    local genome_dir="$1"
    local label="$2"

    if ! "$SINGLIFY" genome status "$genome_dir" 2>/dev/null | grep -q "loaded"; then
        echo "[slurm_genome_prolog] Loading $label genome into shared memory..."
        if "$SINGLIFY" genome load "$genome_dir"; then
            echo "[slurm_genome_prolog] $label genome loaded"
        else
            echo "[slurm_genome_prolog] WARNING: failed to load $label genome — jobs will use per-job loading"
        fi
    else
        echo "[slurm_genome_prolog] $label genome already loaded in shared memory"
    fi
}

# Human (always loaded)
load_genome "${SINGLIFY_REF_BASE}/GRCh38-2024-A/star_2.7.11b" "GRCh38 (human)"

# Mouse (uncomment for mixed-species batches)
# load_genome "${SINGLIFY_REF_BASE}/GRCm39-2024-A/star_2.7.11b" "GRCm39 (mouse)"
