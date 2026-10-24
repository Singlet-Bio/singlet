#!/bin/bash
#SBATCH --job-name=pb-compare
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch compare.sbatch <dataset_or_bam_GSM> [--barnyard] [--cite]
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
export NUMBA_CACHE_DIR=/tmp/$USER-numba MPLCONFIGDIR=/tmp/$USER-mpl
python -I $SRC_NEW/bench/pipeline/analysis/compare.py $PB "$@"
