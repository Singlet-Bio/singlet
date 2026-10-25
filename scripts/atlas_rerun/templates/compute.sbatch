#!/bin/bash
# Atlas re-run compute stage: one array task = one staged sample, the pinned binary with a private genome.
# Chained to the I/O array task by task (--dependency=aftercorr:<io array> --kill-on-invalid-dep=yes): a sample whose
# I/O task failed or parked never starts. Threads = --cpus-per-task; Slurm options come from the controller.
#   sbatch [opts] compute.sbatch <orchestrator.py> <config.json> <cluster> <batch.jsonl>
set -u
ORCH=$1; CONFIG=$2; CLUSTER=$3; BATCH=$4
exec "${AR_PYTHON:-python3}" "$ORCH" compute-stage --config "$CONFIG" --cluster "$CLUSTER" --batch "$BATCH" \
  --index "${SLURM_ARRAY_TASK_ID:-0}" --threads "${SLURM_CPUS_PER_TASK:-4}"
