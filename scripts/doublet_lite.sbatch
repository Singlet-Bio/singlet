#!/bin/bash
# Clipper job for scripts/doublet_lite.py: ARGS="select|fit|feats|eval ..." (the --out is added). numpy/scipy/sklearn only, no torch.
#   sbatch --mem=300G --export=ALL,ARGS="select" scripts/doublet_lite.sbatch
#SBATCH --job-name=dbllite
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
: "${ARGS:?set ARGS}"
BRANCH=${BRANCH:-feat/doublet-score-v2}
SRC=$WORK/src/singlet/${BRANCH//\//__}
OUT=${OUT:-$WORK/doublet_v2}
(
  flock -w 600 9
  if [ -d "$SRC/.git" ]; then git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q -f FETCH_HEAD
  else git clone -q -b "$BRANCH" git@github.com:Singlet-Bio/singlet.git "$SRC"; fi
) 9>"$WORK/state/doublet-v2.lock"
echo "doublet_lite @ $(git -C "$SRC" rev-parse --short HEAD) args=$ARGS host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "PYTHONPATH=$SRC/python:$WORK/venvs/ctnn-torch" --env "SINGLET_CELLTYPE_NN_MODEL=$WORK/models/sha256/b63f5b394de83e8d1523a25a5af55d69a5dbf068381cd3b4ce0f8e4f360ce59b.pt" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
  --env "MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "OMP_WAIT_POLICY=PASSIVE" --env "PYTHONWARNINGS=ignore" \
  --env "LQC=$WORK/latent_qc" "$WORK/images/python-3.12-bookworm.sif" \
  python -u "$SRC/scripts/${SCRIPT:-doublet_lite.py}" $ARGS --out "$OUT"
