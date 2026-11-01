#!/bin/bash
# Singlet's own methods into the comparison's per-cell format (bench.py --save-calls), one study per array task. NAME = model.
#SBATCH --job-name=ctnn-ours
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
B=$WORK/data/annotate/bench; G=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $SRC/scripts/celltype_nn/compete/studies.txt)
mkdir -p $D/compete/ours
in_runtime env CELLTYPIST_FOLDER=$WORK/data/celltypist PYTHONPATH=$PYTHONPATH:$TORCH python -u $SRC/scripts/celltype_nn/bench.py \
  --model $D/models/${NAME:-m1_notissue}/model.pt --data $D --truth $B/truth --bundles $B/bundles --studies $G \
  --methods nn+nn_argmax+guide --thresholds 0.3 --out $D/compete/ours/${NAME:-m1_notissue}_$G.json --save-calls $D/compete/calls_${NAME:-m1_notissue}
