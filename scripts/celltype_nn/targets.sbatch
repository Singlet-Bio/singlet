#!/bin/bash
# Crosswalk, consensus refinements and targets (scripts/celltype_nn/ensemble_run.py targets).
#SBATCH --job-name=ctnn-tgt
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime python -u $SRC/scripts/celltype_nn/ensemble_run.py targets --data $D
