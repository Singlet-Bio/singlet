#!/bin/bash
#SBATCH --job-name=adt-sync
#SBATCH --account=debruinz
#SBATCH --partition=debug-bigmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fetch only this branch into the bare clone and check it out in its own worktree (never sync.sh). Also places the
# inventory where the array jobs read it.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BRANCH=claude/adt-modality-0637a6
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
git -C $SRC checkout -q --detach "$BRANCH"
git -C $SRC log --oneline -1
mkdir -p $WORK/state/adt/lists $WORK/data/adt $WORK/cache/adt
cp $SRC/bench/adt/inventory/adt_hto_inventory.tsv $SRC/bench/adt/inventory/gex_candidates.tsv $WORK/state/adt/ 2>/dev/null || true
cp $SRC/bench/adt/inventory/lists/*.txt $WORK/state/adt/lists/ 2>/dev/null || true
ls -la $WORK/state/adt $WORK/state/adt/lists
