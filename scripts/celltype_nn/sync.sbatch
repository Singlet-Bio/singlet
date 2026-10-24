#!/bin/bash
# Put branch claude/celltype-nn of Singlet-Bio/singlet at $WORK/src/singlet-py/claude__celltype-nn and build its readers.
# Submit before the other celltype_nn jobs (they depend on it with --dependency=afterok).
#SBATCH --job-name=ctnn-sync
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BRANCH=claude/celltype-nn
SRC=$WORK/src/singlet-py/claude__celltype-nn
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
[ -d "$SRC" ] || git -C $WORK/repos/singlet-py.git worktree add -f --detach "$SRC" "$BRANCH"
git -C "$SRC" checkout -q --detach "$BRANCH"
git -C "$SRC" reset -q --hard "$BRANCH"
ls "$SRC"/python/singlet/_pz.*.so >/dev/null 2>&1 || (cd "$SRC" && in_runtime python setup.py -q build_ext --inplace)
git -C "$SRC" log --oneline -1
