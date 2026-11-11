#!/bin/bash
#SBATCH --job-name=dbl-post
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$WORK/src/singlet/calls-sota__doublets
git -C "$SRC" fetch -q origin calls-sota/doublets && git -C "$SRC" checkout -q -f FETCH_HEAD
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" --env "HDF5_USE_FILE_LOCKING=FALSE" "$WORK/images/python-3.12-bookworm.sif" python -u "$SRC/bench/calls-sota/doublets/posthoc.py" "$WORK/bench/calls-sota/doublets"
