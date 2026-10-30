#!/bin/bash
# Atlas-wide cell type probabilities, one shard of studies per array task (scripts/celltype_nn/infer_campaign.py).
# ASK ZACH FIRST: this is the atlas-wide run. Plan (a job, not the login node):
#   python scripts/celltype_nn/infer_campaign.py plan --shards 200 --out $OUT/plan.tsv
# Submit, throttled:
#   sbatch --array=0-199%40 --export=ALL,OUT=$OUT,MODEL=$MODEL,MODEL_VERSION=$MODEL_VERSION scripts/celltype_nn/infer_campaign.sbatch
#SBATCH --job-name=ctnn-infer
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
: "${OUT:?}" "${MODEL:?}" "${MODEL_VERSION:?}"
SCRATCH=/tmp/$USER-ctnn-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT
in_runtime env PYTHONPATH=$PYTHONPATH:$TORCH python -u $SRC/scripts/celltype_nn/infer_campaign.py run --plan $OUT/plan.tsv \
  --shard $SLURM_ARRAY_TASK_ID --model $MODEL --model-version "$MODEL_VERSION" --out $OUT --scratch $SCRATCH \
  --ledger $OUT/ledger/shard-$SLURM_ARRAY_TASK_ID.jsonl
