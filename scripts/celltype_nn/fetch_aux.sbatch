#!/bin/bash
# Auxiliary downloads for celltype_nn: the CellTypist models the ensemble uses, and a few Singlet bundles (not benchmark
# studies) whose spliced/unspliced layers give per-gene intron/exon ratios for the augmentation.
#SBATCH --job-name=ctnn-aux
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
export CELLTYPIST_FOLDER=$WORK/data/celltypist
in_runtime env CELLTYPIST_FOLDER=$CELLTYPIST_FOLDER python - <<'PY'
from singlet.celltype_nn.ensemble import CELLTYPIST_MODELS
import celltypist
ms = sorted({m for sp in CELLTYPIST_MODELS.values() for v in sp.values() for m in v})
celltypist.models.download_models(model=ms, force_update=False)
print("models:", ms)
PY
mkdir -p $D/aux_bundles
for g in ${AUX_STUDIES:-GSE149689 GSE115469 GSE202109}; do
  f=$D/aux_bundles/$g.singlet
  [ -s $f ] || curl -sfL -A "singlet-celltype-nn/0.1" -o $f.part "https://data.singlet.bio/data/$g/$g.singlet" && mv $f.part $f || echo "$g: download failed"
  ls -la $f 2>/dev/null || true
done
