#!/bin/bash
# Ensemble consensus labels on Singlet's own counts, one shard of studies per array task (scripts/celltype_nn/singlet_pseudo.py).
# Plan first: PLAN=1 sbatch scripts/celltype_nn/pseudo.sbatch ; then sbatch --array=0-39%20 scripts/celltype_nn/pseudo.sbatch
#SBATCH --job-name=ctnn-pseudo
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
if [ -n "${PLAN:-}" ]; then
  in_runtime python -u $SRC/scripts/celltype_nn/singlet_pseudo.py plan --data $D --exclude-studies $WORK/jobs/annotate/bench_studies.json \
    --out $D/pseudo/plan.tsv
  exit 0
fi
SCRATCH=/tmp/$USER-ctnn-pseudo-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
in_runtime env CELLTYPIST_FOLDER=$WORK/data/celltypist python -u $SRC/scripts/celltype_nn/singlet_pseudo.py run --data $D \
  --plan $D/pseudo/plan.tsv --shard $SLURM_ARRAY_TASK_ID --out $D/pseudo --scratch $SCRATCH \
  --celltypist-models $WORK/data/celltypist/data/models
