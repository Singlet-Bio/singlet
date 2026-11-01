#!/bin/bash
# Fetch the planned Census cells, one chunk (~50k cells, all genes, raw counts) per array task:
#   sbatch --array=0-$((NCHUNKS-1))%12 scripts/celltype_nn/corpus_fetch.sbatch
#SBATCH --job-name=ctnn-fetch
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime env PYTHONPATH=$CENSUS:$PYTHONPATH python -u $SRC/scripts/celltype_nn/corpus.py fetch --out $D --chunk $SLURM_ARRAY_TASK_ID
