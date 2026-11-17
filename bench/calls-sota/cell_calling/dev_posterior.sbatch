#!/bin/bash
#SBATCH --job-name=cc-devpost
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
T=$W/state/cellcall_truth/truth; O=$W/bench/calls-sota/cell_calling; mkdir -p $O
$W/envs/gpool/venv/bin/python -P $O/score_posterior.py $T $O/dev_scores_posterior.json \
  kang_b1A:G kang_b1B:G kang_b1C:G kang_b2ctrl:G hgmm1k_v2:S hgmm1k_v3:S hgmm5k_v3:S
