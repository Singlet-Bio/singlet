#!/bin/bash
# Run scripts/restrict_published_bundles.py on Clipper. One array task is one shard of the studies (accession order).
#
#   dry    read-only: counts per study (no R2 writes)
#   pilot  GSE128639 and the two smallest other affected studies, with writes
#   full   every shard of NSHARD with writes; safe to resubmit, studies already split are skipped
#
#   sbatch --array=0-7%8 --export=ALL,MODE=full,COMMIT=<sha>  scripts/restrict_published_bundles.sbatch
#   sbatch --array=0     --export=ALL,MODE=pilot,COMMIT=<sha> scripts/restrict_published_bundles.sbatch
#
# Needs ~/.config/singlet-r2.env (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY) and $WORK/venvs/r2tools (boto3).
#SBATCH --job-name=restrict
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
MODE=${MODE:-dry}
NSHARD=${NSHARD:-8}
TASK=${SLURM_ARRAY_TASK_ID:-0}
COMMIT=${COMMIT:?set COMMIT to the exact singlet-bio commit to run (git rev-parse origin/feat/restrict-genotype-members)}
SRC=$WORK/src/singlet-py/restrict__${COMMIT:0:10}
PY=$WORK/venvs/r2tools/bin/python
PRIVATE=${PRIVATE:-singlet-restricted}
LEDGER=$WORK/state/restrict-ledger-$MODE-$TASK.jsonl

mkdir -p "$WORK/state"
# The code under test is pinned to one commit, checked out once (the array tasks share it).
(
  flock 9
  if [ ! -e "$SRC" ]; then
    git --git-dir="$WORK/repos/singlet-py.git" fetch -q origin "+refs/heads/feat/restrict-genotype-members:refs/heads/feat/restrict-genotype-members"
    git --git-dir="$WORK/repos/singlet-py.git" worktree add -q --detach "$SRC" "$COMMIT"
  fi
) 9>"$WORK/state/restrict.lock"

set -a; . "$HOME/.config/singlet-r2.env"; set +a
SCRATCH=/tmp/$USER-${SLURM_JOB_ID:-x}-$TASK
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

echo "mode=$MODE shard=$TASK/$NSHARD commit=$(git -C "$SRC" rev-parse --short HEAD) scratch=$SCRATCH"
RUN=("$PY" "$SRC/scripts/restrict_published_bundles.py" migrate --private "$PRIVATE" --ledger "$LEDGER" --scratch "$SCRATCH")

case "$MODE" in
  dry)   "${RUN[@]}" --all --shard "$TASK/$NSHARD" ;;
  pilot) "${RUN[@]}" --gse GSE128639 --apply
         "${RUN[@]}" --all --limit 2 --apply ;;
  full)  "${RUN[@]}" --all --shard "$TASK/$NSHARD" --apply ;;
  *)     echo "MODE must be dry, pilot or full"; exit 2 ;;
esac
