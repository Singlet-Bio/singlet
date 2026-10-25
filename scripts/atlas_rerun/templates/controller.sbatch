#!/bin/bash
# Atlas re-run controller: ticks every interval_s for the life of this job, after queueing its successor
# (--dependency=afterany:<this job>) so the loop survives the partition's time limit. Stop it with a STOP file in the
# campaign root (no successor is queued once STOP exists); pause submissions with a PAUSE file.
# First start (from the laptop, through the clipper skill):
#   sbatch --account=debruinz --partition=short-cpu --cpus-per-task=1 --mem=4G --time=03:55:00 --job-name=ar-ctl-clipper \
#     --output=<root>/slurm/%x-%j.out controller.sbatch <orchestrator.py> <config.json> clipper
set -u
ORCH=$1; CONFIG=$2; CLUSTER=$3; shift 3
exec "${AR_PYTHON:-python3}" "$ORCH" controller --config "$CONFIG" --cluster "$CLUSTER" "$@"
