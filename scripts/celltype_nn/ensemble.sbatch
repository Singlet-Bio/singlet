#!/bin/bash
# Ensemble members on the subset cells of one chunk per array task (scripts/celltype_nn/ensemble_run.py run).
#SBATCH --job-name=ctnn-ens
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime env CELLTYPIST_FOLDER=$WORK/data/celltypist python -u $SRC/scripts/celltype_nn/ensemble_run.py run --data $D   --chunk $SLURM_ARRAY_TASK_ID --celltypist-models $WORK/data/celltypist/data/models
