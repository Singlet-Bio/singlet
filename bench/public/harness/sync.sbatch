#!/bin/bash
#SBATCH --job-name=bpub-sync
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Bring this branch's Clipper worktree to the pushed tip (only this branch is fetched into the shared bare clone).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BR=claude/bench-public-cohort-485cf7; BARE=$WORK/repos/singlet-py.git; SRC=$WORK/src/singlet-py/${BR//\//__}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
git -C $SRC reset -q --hard refs/heads/$BR; git -C $SRC log --oneline -1
