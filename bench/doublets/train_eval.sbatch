#!/bin/bash
# Train the packaged combiner and write the leave-one-study-out artifacts for bench/public/capabilities/doublets.py.
# Inputs are the cached scores of the latent-qc study ($WORK/latent_qc/dbl); the code is a synced checkout of this branch (SRC).
# Submit: sbatch --export=ALL,SRC=$SRC bench/doublets/train_eval.sbatch
#SBATCH --job-name=dbl-train
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=${SRC:-$WORK/jobs/dbl-train}
OUT=${OUT:-$WORK/bench-public/artifacts/doublets}
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
mkdir -p "$OUT"
export PYTHONPATH=$SRC/bench/doublets:$SRC/python OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
in_runtime python -u $SRC/bench/doublets/train_eval.py "$OUT"
