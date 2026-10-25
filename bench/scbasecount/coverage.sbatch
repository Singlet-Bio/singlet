#!/bin/bash
#SBATCH --job-name=sbc-cov
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
D=$WORK/jobs/sbc
in_runtime python $D/coverage.py $WORK/data/scbasecount/meta $WORK/state/v2/plan-09fd92b149/plan.tsv $WORK/data/scbasecount/coverage
