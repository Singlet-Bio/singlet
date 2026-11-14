#!/bin/bash
#SBATCH --job-name=cc-devrank
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/bench/calls-sota/cell_calling
NUMBA_NUM_THREADS=8 NUMBA_CACHE_DIR=/tmp/$USER-nb-$SLURM_JOB_ID $W/envs/gpool/venv/bin/python -P $S/dev_rank.py $S/cellcall_v22_rank.py $S/dev_rank.json
