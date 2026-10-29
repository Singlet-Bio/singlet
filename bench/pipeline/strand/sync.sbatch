#!/bin/bash
#SBATCH --job-name=pb-strand-sync
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fetch ONLY fix/strand-detection into the shared bare repo and refresh its worktree.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
REPO=$WORK/repos/singlet-py.git
git -C $REPO fetch --update-head-ok origin +refs/heads/fix/strand-detection:refs/heads/fix/strand-detection
W=$WORK/src/singlet-py/fix__strand-detection
[ -d $W ] || git -C $REPO worktree add $W fix/strand-detection
git -C $W reset -q --hard refs/heads/fix/strand-detection
git -C $W log --oneline -1
