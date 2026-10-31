#!/bin/bash
# Clipper jobs of the doublet/score_v2 stacker (scripts/doublet_v2.py; docs/cell_features.md section 8). Pick the work with MODE:
#   train      fit the frozen stacker -> $OUT/model/doublet_v2_stacker.{npz,json} (commit them to python/singlet/features/data/)
#   feats      atlas + sdfl inputs recomputed by the package for the 17 datasets ($OUT/pkgfeat, cost.json)
#   train-pkg  frozen stacker fitted on those recomputed inputs ($OUT/model_pkg)
#   loso       (FEATURES=stored|pkg|cross) package leave-one-study-out scores + mean AUPRC ($OUT/loso_pkg)
#   ref-loso   stack.py recipe, scores saved ($OUT/loso_ref)
#   score      cost + agreement of the whole package on one dataset: NAME=pbmc-1A-dm (DEVICE=cpu|cuda)
#   pytest     tests/python/test_doublet_v2.py + test_cell_features.py with the data mounted
# Code: a clone of BRANCH (default feat/doublet-score-v2) under $WORK/src/singlet. Real data only: $LQC/dbl.
#   sbatch --export=ALL,MODE=score,NAME=kang21 scripts/doublet_v2.sbatch
#SBATCH --job-name=dblv2
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
: "${MODE:?set MODE}"
BRANCH=${BRANCH:-feat/doublet-score-v2}
SRC=$WORK/src/singlet/${BRANCH//\//__}
OUT=${OUT:-$WORK/doublet_v2}
LQC=$WORK/latent_qc
mkdir -p "$OUT" "$WORK/state" "$WORK/data/doublet_v2"
(
  flock -w 600 9
  if [ -d "$SRC/.git" ]; then git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q -f FETCH_HEAD
  else git clone -q -b "$BRANCH" git@github.com:Singlet-Bio/singlet.git "$SRC"; fi
) 9>"$WORK/state/doublet-v2.lock"
echo "doublet_v2 @ $(git -C "$SRC" rev-parse --short HEAD) mode=$MODE host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
CEN=$WORK/data/doublet_v2/atlas_centroids.npz
[ -e "$CEN" ] || cp "$LQC/dbl/atlas_centroids.npz" "$CEN"
export SINGLET_CELLTYPE_NN_MODEL=$WORK/data/celltype_nn/models/m1_notissue/model.pt SINGLET_DOUBLET_CENTROIDS=$CEN
export SINGLET_DBL_REF=$OUT LQC=$LQC
rt() {
  apptainer exec ${NV:+--nv} --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "PYTHONPATH=$SRC/python:$WORK/venvs/ctnn-torch" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "HDF5_USE_FILE_LOCKING=FALSE" \
    --env "PYTHONWARNINGS=ignore" --env "OMP_WAIT_POLICY=PASSIVE" --env "SINGLET_CELLTYPE_NN_MODEL=$SINGLET_CELLTYPE_NN_MODEL" \
    --env "SINGLET_DOUBLET_CENTROIDS=$SINGLET_DOUBLET_CENTROIDS" --env "SINGLET_DBL_REF=$OUT" --env "LQC=$LQC" \
    "$WORK/images/python-3.12-bookworm.sif" "$@"
}
[ "${DEVICE:-cpu}" = cuda ] && NV=1
case "$MODE" in
  train)    rt python -u "$SRC/scripts/doublet_v2.py" train --model-dir "$OUT/model" ;;
  feats)    rt python -u "$SRC/scripts/doublet_v2.py" feats --out "$OUT" --device "${DEVICE:-cpu}" ;;
  loso)     rt python -u "$SRC/scripts/doublet_v2.py" loso --out "$OUT" --features "${FEATURES:-stored}" --pkgdir "$OUT/pkgfeat" ;;
  train-pkg) rt python -u "$SRC/scripts/doublet_v2.py" train --model-dir "$OUT/model_pkg" --features pkg --pkgdir "$OUT/pkgfeat" ;;
  ref-loso) rt python -u "$SRC/scripts/doublet_v2.py" ref-loso --out "$OUT" ;;
  score)    /usr/bin/time -v apptainer exec ${NV:+--nv} --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
              --env "PYTHONPATH=$SRC/python:$WORK/venvs/ctnn-torch" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
              --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "PYTHONWARNINGS=ignore" --env "OMP_WAIT_POLICY=PASSIVE" \
              --env "SINGLET_CELLTYPE_NN_MODEL=$SINGLET_CELLTYPE_NN_MODEL" --env "SINGLET_DOUBLET_CENTROIDS=$SINGLET_DOUBLET_CENTROIDS" \
              --env "LQC=$LQC" "$WORK/images/python-3.12-bookworm.sif" \
              python -u "$SRC/scripts/doublet_v2.py" score "${NAME:?set NAME}" --out "$OUT/score" --device "${DEVICE:-cpu}" ;;
  pytest)   cd "$SRC" && rt python -m pytest -q tests/python/test_doublet_v2.py tests/python/test_cell_features.py -p no:cacheprovider ;;
  *) echo "unknown MODE $MODE" >&2; exit 2 ;;
esac
