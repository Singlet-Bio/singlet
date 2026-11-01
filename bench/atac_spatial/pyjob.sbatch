#!/bin/bash
#SBATCH --job-name=as-py
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run one Python script of bench/atac_spatial (synced branch worktree) in the atacbench env: sbatch pyjob.sbatch SCRIPT ARGS...
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
BARE=$W/repos/singlet-py.git; BR=claude/atac-spatial-benchmark-2ba7df
SRC=$W/src/singlet-py/claude__atac-spatial-benchmark-2ba7df__scripts   # scripts only; the build job owns the other worktree
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q --detach $SRC refs/heads/$BR
git -C $SRC checkout -q --detach refs/heads/$BR; git -C $SRC log --oneline -1
PY=${PY:-$W/envs/atacbench/bin/python}
[ -x $PY ] || PY=$W/envs/adt-bench/bin/python
$PY -c "import pyliftover" 2>/dev/null || $(dirname $PY)/pip install -q pyliftover
S=$1; shift
$PY $SRC/bench/atac_spatial/$S "$@"
