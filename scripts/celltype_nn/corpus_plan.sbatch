#!/bin/bash
# Plan the Census training corpus (scripts/celltype_nn/corpus.py plan) and fetch the Ensembl orthologs. Reads Census obs of
# every primary human and mouse cell (about 100M rows), so it needs a large-memory node.
#SBATCH --job-name=ctnn-plan
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
mkdir -p $D
in_runtime env PYTHONPATH=$CENSUS:$PYTHONPATH python -u $SRC/scripts/celltype_nn/corpus.py orthologs --out $D
in_runtime env PYTHONPATH=$CENSUS:$PYTHONPATH python -u $SRC/scripts/celltype_nn/corpus.py plan --out $D \
  --exclude-studies $WORK/jobs/annotate/bench_studies.json ${PLAN_ARGS:-}
