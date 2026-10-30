#!/bin/bash
# Shared inputs of the comparison (export.py reference + queries), then CellTypist training on the corpus.
#SBATCH --job-name=ctnn-cprep
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
B=$WORK/data/annotate/bench; O=$D/compete
in_runtime python -u $SRC/scripts/celltype_nn/compete/export.py reference --data $D --out $O
in_runtime python -u $SRC/scripts/celltype_nn/compete/export.py queries --data $D --truth $B/truth --bundles $B/bundles --out $O
in_runtime python -u $SRC/scripts/celltype_nn/compete/run_py.py train --method celltypist --data $D --out $O
