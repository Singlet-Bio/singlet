#!/bin/bash
# Vocabulary, projection of every chunk, intron/exon ratios (scripts/celltype_nn/prepare.py).
#SBATCH --job-name=ctnn-prep
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
B=$WORK/data/annotate/bench/bundles
AUX=$(ls $D/aux_bundles/*.singlet | paste -sd, -)
in_runtime python -u $SRC/scripts/celltype_nn/prepare.py --data $D --singlet-genes human=$B/GSE130973.singlet,mouse=$B/GSE157480.singlet   --intron-bundles "$AUX"
in_runtime python -u $SRC/scripts/celltype_nn/ensemble_run.py space --data $D
