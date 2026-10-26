#!/bin/bash
#SBATCH --job-name=vdjc-py
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run one script of this branch ($PY, repo-relative) with $ARGS in the env.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
python -P $SRC/$PY ${ARGS:-}
