#!/bin/bash
# Train on one GPU (scripts/celltype_nn/train.py). NAME and TRAIN_ARGS from the environment.
#SBATCH --job-name=ctnn-train
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime_gpu env PYTHONPATH=$TORCH:$PYTHONPATH python -u $SRC/scripts/celltype_nn/train.py --data $D --out $D/models/${NAME:-m1}   --workers 12 ${TRAIN_ARGS:-}
