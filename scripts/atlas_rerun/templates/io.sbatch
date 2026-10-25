#!/bin/bash
# Atlas re-run I/O stage: one array task = one sample of a batch file.
#   head screen of every run (1M read pairs, streamed) -> prefetch of each kept run -> local decode -> full screen -> pool
# Slurm options (partition, cores, memory, time, --array with its throttle, --requeue) come from the controller's
# sbatch command line (orchestrator.py, Controller.submit_one_batch), so this file is the same on every cluster.
#   sbatch [opts] io.sbatch <orchestrator.py> <config.json> <cluster> <batch.jsonl>
set -u
ORCH=$1; CONFIG=$2; CLUSTER=$3; BATCH=$4
exec "${AR_PYTHON:-python3}" "$ORCH" io-stage --config "$CONFIG" --cluster "$CLUSTER" --batch "$BATCH" \
  --index "${SLURM_ARRAY_TASK_ID:-0}"
