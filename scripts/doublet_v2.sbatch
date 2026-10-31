#!/bin/bash
# Clipper jobs of the doublet/score_v2 stacker (scripts/doublet_v2.py; docs/cell_features.md section 8). Pick the work with MODE:
#   feats      the package's inputs for the 17 evaluation datasets -> $OUT/pkgfeat (+ cost.json)
#   train      fit the frozen stacker on them -> $OUT/model (copy to python/singlet/features/data/)
#   loso       leave-one-study-out scores of the package + mean AUPRC and CI against scDblFinder ($OUT/loso_pkg)
#   ref-loso   the stack.py recipe on the same inputs ($OUT/loso_ref)
#   score      cost of the whole stage on one dataset: NAME=pbmc-1A-dm ($OUT/score)
#   pytest     tests/python/test_doublet_v2.py + test_cell_features.py with the data mounted
# Code: a clone of BRANCH (default feat/doublet-score-v2) under $WORK/src/singlet. Real data only: $LQC/dbl. No torch.
#   sbatch --export=ALL,MODE=score,NAME=kang21 scripts/doublet_v2.sbatch
#SBATCH --job-name=dblv2
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
: "${MODE:?set MODE}"
BRANCH=${BRANCH:-feat/doublet-score-v2}
SRC=$WORK/src/singlet/${BRANCH//\//__}
OUT=${OUT:-$WORK/doublet_v2}
LQC=$WORK/latent_qc
mkdir -p "$OUT" "$WORK/state"
(
  flock -w 600 9
  if [ -d "$SRC/.git" ]; then git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q -f FETCH_HEAD
  else git clone -q -b "$BRANCH" git@github.com:Singlet-Bio/singlet.git "$SRC"; fi
) 9>"$WORK/state/doublet-v2.lock"
echo "doublet_v2 @ $(git -C "$SRC" rev-parse --short HEAD) mode=$MODE host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
rt() {
  apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "PYTHONPATH=$SRC/python" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
    --env "MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "HDF5_USE_FILE_LOCKING=FALSE" --env "PYTHONWARNINGS=ignore" \
    --env "OMP_WAIT_POLICY=PASSIVE" --env "SINGLET_DBL_REF=$OUT" --env "LQC=$LQC" "$WORK/images/python-3.12-bookworm.sif" "$@"
}
case "$MODE" in
  feats)    rt python -u "$SRC/scripts/doublet_v2.py" feats --out "$OUT" ;;
  train)    rt python -u "$SRC/scripts/doublet_v2.py" train --out "$OUT" --model-dir "$OUT/model" ;;
  loso)     rt python -u "$SRC/scripts/doublet_v2.py" loso --out "$OUT" ;;
  ref-loso) rt python -u "$SRC/scripts/doublet_v2.py" ref-loso --out "$OUT" ;;
  score)    /usr/bin/time -v apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
              --env "PYTHONPATH=$SRC/python" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
              --env "PYTHONWARNINGS=ignore" --env "OMP_WAIT_POLICY=PASSIVE" --env "LQC=$LQC" "$WORK/images/python-3.12-bookworm.sif" \
              python -u "$SRC/scripts/doublet_v2.py" score "${NAME:?set NAME}" --out "$OUT/score" ;;
  pytest)   cd "$SRC" && rt python -m pytest -q tests/python/test_doublet_v2.py tests/python/test_cell_features.py -p no:cacheprovider ;;
  *) echo "unknown MODE $MODE" >&2; exit 2 ;;
esac
