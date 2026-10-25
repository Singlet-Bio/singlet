#!/bin/bash
# scVI + scANVI on the comparison's reference (one model per species), one GPU.
#SBATCH --job-name=ctnn-scanvi
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime_gpu env PYTHONPATH=$TORCH:$WORK/venvs/forge-scvi/lib/python3.12/site-packages:$PYTHONPATH \
  python -u $SRC/scripts/celltype_nn/compete/run_py.py train --method scanvi --data $D --out $D/compete --epochs 100
