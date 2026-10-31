#!/bin/bash
#SBATCH --job-name=vdjh-score
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Score runs with the pre-registered primary metric (PREREG.md).
#   sbatch --export=ALL,LIBS=ctrl10k_t:TCR,ctrl10k_b:BCR,...,OUT=name[,SDIR=singlet] score.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=bench/vdj-heldout
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
mkdir -p $WORK/state/vdjh
python -P $SRC/bench/expression_sota/vdj/primary.py --runs ${RUNS:-$WORK/state/vdjc/runs} --data $WORK/data/vdjc \
  --libs $LIBS --singlet-dir ${SDIR:-singlet} --out $WORK/state/vdjh/$OUT.json
