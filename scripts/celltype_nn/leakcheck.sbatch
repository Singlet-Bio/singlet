#!/bin/bash
# Are the benchmark collections' cells Census primary data (so that no copy of them is in the training corpus)?
#SBATCH --job-name=ctnn-leak
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
in_runtime env PYTHONPATH=$CENSUS:$PYTHONPATH python -u $SRC/scripts/celltype_nn/corpus.py leakcheck --out $D --exclude-studies $WORK/jobs/annotate/bench_studies.json
