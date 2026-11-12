#!/bin/bash
#SBATCH --job-name=cc-diag5kv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/bench/calls-sota/cell_calling
ls $W/bench-public/runs/hgmm5k_v3/singlet | head -30
NUMBA_NUM_THREADS=4 NUMBA_CACHE_DIR=/tmp/$USER-nb-$SLURM_JOB_ID $W/envs/gpool/venv/bin/python -P $S/diag_hgmm5k_variants.py $S/cellcall_v21.py $W/bench-public/runs/hgmm5k_v3/singlet $W/state/cellcall_truth/truth
