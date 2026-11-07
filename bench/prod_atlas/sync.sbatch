#!/bin/bash
#SBATCH --job-name=ar-syncprod
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu,short-cpu,scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Move the prod/atlas-rerun worktree to the branch tip (scripts only; the binary is rebuilt by build.sbatch).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git; SRC=$W/src/singlet-py/prod__atlas-rerun
git -C $BARE fetch -q --update-head-ok origin +refs/heads/prod/atlas-rerun:refs/heads/prod/atlas-rerun
[ -d $SRC ] || git -C $BARE worktree add -q $SRC prod/atlas-rerun
git -C $SRC reset -q --hard refs/heads/prod/atlas-rerun; git -C $SRC log --oneline -1
