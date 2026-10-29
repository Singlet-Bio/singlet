#!/bin/bash
# Benchmark on Singlet counts (scripts/celltype_nn/bench.py). NAME (model) and BENCH_ARGS from the environment.
#SBATCH --job-name=ctnn-bench
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
B=$WORK/data/annotate/bench
mkdir -p $D/bench
in_runtime env CELLTYPIST_FOLDER=$WORK/data/celltypist PYTHONPATH=$PYTHONPATH:$TORCH python -u $SRC/scripts/celltype_nn/bench.py   --model $D/models/${NAME:-m1}/model.pt --data $D --truth $B/truth --bundles $B/bundles   --celltypist-models $WORK/data/celltypist/data/models --out $D/bench/${NAME:-m1}${TAG:-}.json   --save-probs $D/bench/${NAME:-m1}${TAG:-}.cells.parquet ${BENCH_ARGS:-}
in_runtime python $SRC/scripts/celltype_nn/bench_summary.py $D/bench/${NAME:-m1}${TAG:-}.json --json $D/bench/${NAME:-m1}${TAG:-}.summary.json > $D/bench/${NAME:-m1}${TAG:-}.md
