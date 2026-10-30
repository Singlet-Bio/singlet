#!/bin/bash
# Refit a model's temperature on call reliability (scripts/celltype_nn/calibrate.py). NAME = model directory.
#SBATCH --job-name=ctnn-calib
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime_gpu env PYTHONPATH=$TORCH:$PYTHONPATH python -u $SRC/scripts/celltype_nn/calibrate.py --data $D --model $D/models/${NAME:-m1} ${CALIB_ARGS:-}
