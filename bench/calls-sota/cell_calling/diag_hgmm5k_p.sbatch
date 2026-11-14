#!/bin/bash
#SBATCH --job-name=cc-diag5kp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/gpool/venv/bin/python -P $W/bench/calls-sota/cell_calling/diag_hgmm5k_p.py $W/state/cellcall_truth/truth/hgmm5k_v3.truth.tsv.gz $W/state/cellcall_truth/cv2/hgmm5k_v3.tsv
