#!/bin/bash
#SBATCH --job-name=vdj2-py
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run a script of this branch (pushed commit) in the vdjcrispr env, once per ';'-separated argument set, from
# $WORK/expression-sota/vdj2:   sbatch --export=ALL,SCRIPT=bench/..py,PYARGS="a b;c d" py.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-bench/vdj-sota}
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
mkdir -p $WORK/expression-sota/vdj2 && cd $WORK/expression-sota/vdj2
IFS=";" read -ra SETS <<< "$PYARGS"
for x in "${SETS[@]}"; do eval python -P $SRC/$SCRIPT $x || echo "FAILED: $x"; done
