#!/bin/bash
#SBATCH --job-name=pb-sync
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fetch ONLY fix/molecule-labels into the shared bare repo and refresh this study's two
# worktrees (fixed branch + baseline at b1335a8). Never runs the global sync.sh.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
REPO=$WORK/repos/singlet-py.git
git -C $REPO fetch origin +refs/heads/fix/molecule-labels:refs/heads/fix/molecule-labels
NEW=$WORK/src/singlet-py/fix__molecule-labels
OLD=$WORK/src/singlet-py/pb-baseline
[ -d $NEW ] || git -C $REPO worktree add $NEW fix/molecule-labels
git -C $NEW checkout -q --detach && git -C $NEW checkout -q fix/molecule-labels 2>/dev/null || true
git -C $NEW reset -q --hard refs/heads/fix/molecule-labels
[ -d $OLD ] || git -C $REPO worktree add --detach $OLD b1335a8
git -C $NEW log --oneline -1; git -C $OLD log --oneline -1
