#!/bin/bash
# Held-out doublet scoring (gate: bench/calls-sota/gates/doublets.md). sbatch --export=ALL,NAME=cellplex10k run_heldout.sbatch
#SBATCH --job-name=dbl-held
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
: "${NAME:?set NAME}"
BRANCH=${BRANCH:-calls-sota/doublets}
SRC=$WORK/src/singlet/${BRANCH//\//__}
OUT=$WORK/bench/calls-sota/doublets
mkdir -p "$OUT"
(
  flock -w 600 9
  if [ -d "$SRC/.git" ]; then git -C "$SRC" fetch -q origin "$BRANCH" && git -C "$SRC" checkout -q -f FETCH_HEAD
  else git clone -q -b "$BRANCH" git@github.com:Singlet-Bio/singlet.git "$SRC"; fi
) 9>"$WORK/state/calls-sota-doublets.lock"
echo "calls-sota doublets @ $(git -C "$SRC" rev-parse --short HEAD) NAME=$NAME host=$(hostname)"
(
  flock -w 900 8
if [ ! -x "$OUT/dbl_run" ]; then
  CXX=$WORK/envs/pbench/bin/x86_64-conda-linux-gnu-g++
  $CXX -O2 -std=c++17 -pthread -static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive -I"$SRC/include/singlet/pileup" "$SRC/bench/modalities/doublet_fix/dbl_run.cpp" -o "$OUT/dbl_run"
  git -C "$SRC" rev-parse HEAD > "$OUT/dbl_run.commit"
fi
) 8>"$WORK/state/calls-sota-dbl-build.lock"
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "PYTHONPATH=$SRC/python" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK" \
  --env "HDF5_USE_FILE_LOCKING=FALSE" --env "PYTHONWARNINGS=ignore" "$WORK/images/python-3.12-bookworm.sif" \
  python -u "$SRC/bench/calls-sota/doublets/run_heldout.py" "$NAME" "$OUT" "$OUT/dbl_run" "$WORK/envs/ambient/bin/Rscript" "$SRC/bench/calls-sota/doublets/scdblfinder.R"
