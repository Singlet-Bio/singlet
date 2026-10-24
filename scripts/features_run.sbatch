#!/bin/bash
# Per-cell feature sidecars on Clipper: one array task = every n-th study of a list (scripts/features_run.py run --shard).
#
# Code: SRC = a checkout of this branch. Until singlet.celllabels (the label readers) and singlet.features share a branch, set
# LABELS_SRC to a synced worktree of the annotation branch with its compiled readers (python setup.py build_ext --inplace),
# e.g. $WORK/src/singlet-py/claude__v2-cell-annotation-6e50dc; the job overlays singlet/features onto a copy of it in scratch.
# Without SRC, the job clones BRANCH (default claude/v2-cell-features) from GitHub into scratch.
# Submit (ASK ZACH FIRST for an atlas-wide run), e.g. 200 shards, 60 at a time:
#   sbatch --array=0-199%60 --export=ALL,OUT=$OUT,LIST=$OUT/studies.txt,SHARDS=200,LABELS_SRC=... scripts/features_run.sbatch
# Afterwards (a job): features_run.py registry --out $OUT; features_run.py collect --out $OUT
#
# Sizing (pilot, docs/cell_features.md): scoring takes 2-10 s per 10k cells on one core; reading counts takes longer.
#SBATCH --job-name=features
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
: "${OUT:?set OUT}" "${LIST:?set LIST}"
SHARDS=${SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh   # pinned image + venv
SCRATCH=/tmp/$USER-features-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}-${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT   # downloaded bundles live only here
if [ -z "${SRC:-}" ]; then
  SRC=$SCRATCH/src
  git clone -q --depth 1 --branch "${BRANCH:-claude/v2-cell-features}" git@github.com:Singlet-Bio/singlet.git "$SRC"
fi
PY=$SRC/python
if [ -n "${LABELS_SRC:-}" ]; then
  PY=$SCRATCH/py; mkdir -p "$PY"; cp -r "$LABELS_SRC/python/singlet" "$PY/"; rm -rf "$PY/singlet/features"
  cp -r "$SRC/python/singlet/features" "$PY/singlet/"
fi
export PYTHONPATH=$PY SINGLET_CACHE_DIR=$SCRATCH/cache OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore
mkdir -p "$OUT"
in_runtime python -u $SRC/scripts/features_run.py run --list "$LIST" --shard ${SLURM_ARRAY_TASK_ID:-0} --shards $SHARDS \
  --out "$OUT" --scratch "$SCRATCH/bundles" --ledger "$OUT/ledger/shard-${SLURM_ARRAY_TASK_ID:-0}.jsonl"
