#!/bin/bash
#SBATCH --job-name=cc-diag5k
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/gpool/venv/bin/python -P $W/bench/calls-sota/cell_calling/diag_hgmm5k.py $W/state/cellcall_truth/truth $W/state/cellcall_truth/cv2/hgmm5k_v3.params.json
