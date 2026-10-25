#!/bin/bash
# Controlled runtime of Singlet's network and CellTypist (scripts/celltype_nn/compete/timing.py) on one exclusive node.
#SBATCH --job-name=ctnn-time
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-bigmem
#SBATCH --exclusive
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime env PYTHONPATH=$PYTHONPATH:$TORCH python -u $SRC/scripts/celltype_nn/compete/timing.py --compete $D/compete \
  --model $D/models/${NAME:-m1_notissue}/model.pt --threads 8
