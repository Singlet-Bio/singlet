#!/bin/bash
#SBATCH --job-name=pb-strand-compare
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# compare.py with singlet-strandfix included; writes results/<dataset>__strand
#   sbatch compare.sbatch <dataset> [--cite]
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__strand-detection/bench/pipeline/strand/common.sh
export NUMBA_CACHE_DIR=/tmp/$USER-numba MPLCONFIGDIR=/tmp/$USER-mpl
python -I $SRC_STRAND/bench/pipeline/analysis/compare.py $PB "$@" --out-suffix __strand
