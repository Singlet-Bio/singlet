#!/bin/bash
# The network on the authors' CELLxGENE matrices of the benchmark cells (scripts/celltype_nn/bench_cxg.py). NAME = model.
#SBATCH --job-name=ctnn-bcxg
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
SCRATCH=/tmp/$USER-ctnn-cxg-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
B=$WORK/data/annotate/bench
in_runtime env PYTHONPATH=$PYTHONPATH:$TORCH python -u $SRC/scripts/celltype_nn/bench_cxg.py --model $D/models/${NAME:-m1}/model.pt \
  --truth $B/truth --studies $WORK/jobs/annotate/bench_studies.json --scratch $SCRATCH --out $D/bench/${NAME:-m1}.cxg.json
in_runtime python $SRC/scripts/celltype_nn/bench_summary.py $D/bench/${NAME:-m1}.cxg.json --json $D/bench/${NAME:-m1}.cxg.summary.json > $D/bench/${NAME:-m1}.cxg.md
