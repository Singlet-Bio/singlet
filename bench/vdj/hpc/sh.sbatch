#!/bin/bash
#SBATCH --job-name=vdjc-sh
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run $CMD (small shell steps: downloads, greps) in the env.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$WORK/envs/vdjcrispr/bin:$PATH HOME=$WORK/cache/condahome
eval "$CMD"
