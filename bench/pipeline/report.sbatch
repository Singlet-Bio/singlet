#!/bin/bash
#SBATCH --job-name=pb-report
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Figures + markdown tables from every results/<dataset>/metrics.json -> results/report/
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
export MPLCONFIGDIR=/tmp/$USER-mpl
python -I $SRC_NEW/bench/pipeline/analysis/make_report.py $PB $PB/results/report
