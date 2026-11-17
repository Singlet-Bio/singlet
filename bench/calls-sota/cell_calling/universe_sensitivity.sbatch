#!/bin/bash
#SBATCH --job-name=cc-univ
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$W/bench/calls-sota/cell_calling
NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/$USER-nb-$SLURM_JOB_ID $W/envs/gpool/venv/bin/python -P $SRC/universe_sensitivity.py $SRC/cellcall_v21.py $SRC/universe_sensitivity.json $SRC/heldout_cellplex10k $SRC/heldout_nuc
