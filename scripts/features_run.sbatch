#!/bin/bash
# Per-cell feature sidecars on Clipper: one array task = every n-th study of a list (scripts/features_run.py run --shard).
#
# Code: SRC = a synced worktree of this branch with the compiled readers built in place (python setup.py build_ext
# --inplace). LABELS_SRC (optional) = a worktree that has the compiled readers when SRC does not: the job copies its
# python/singlet to scratch and puts this branch's singlet/features and singlet/celllabels on top.
# Without SRC, the job clones BRANCH (default claude/v2-cell-features) from GitHub into scratch.
# Submit (ASK ZACH FIRST for an atlas-wide run), e.g. 200 shards, 60 at a time:
#   sbatch --array=0-199%60 --export=ALL,OUT=$OUT,LIST=$OUT/studies.txt,SHARDS=200,SRC=$SRC scripts/features_run.sbatch
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
  PY=$SCRATCH/py; mkdir -p "$PY"; cp -r "$LABELS_SRC/python/singlet" "$PY/"
  rm -rf "$PY/singlet/features" "$PY/singlet/celllabels"
  cp -r "$SRC/python/singlet/features" "$SRC/python/singlet/celllabels" "$PY/singlet/"
fi
export PYTHONPATH=$PY SINGLET_CACHE_DIR=$SCRATCH/cache OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore
mkdir -p "$OUT"
in_runtime python -u $SRC/scripts/features_run.py run --list "$LIST" --shard ${SLURM_ARRAY_TASK_ID:-0} --shards $SHARDS \
  --out "$OUT" --scratch "$SCRATCH/bundles" --ledger "$OUT/ledger/shard-${SLURM_ARRAY_TASK_ID:-0}.jsonl"
