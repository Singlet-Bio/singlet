#!/bin/bash
#SBATCH --job-name=alpha-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Research Alpha benchmark A: one array task per data row of bench/alpha/picks.tsv (task i = row i).
#   sbatch --array=0-N%16 alpha_compare.sbatch [GIT_REF]
# Code comes from GitHub (branch claude/alpha-bench, cloned into node-local scratch); results go to
# $WORK/alpha_bench/compare/<GSM>.json and <GSM>.cells.csv.gz.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
REF=${1:-claude/alpha-bench}
SCR=/tmp/$USER-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git clone -q --depth 1 -b $REF https://github.com/Singlet-Bio/singlet.git $SCR/src 2>/dev/null || \
  git clone -q --depth 1 -b $REF git@github.com:Singlet-Bio/singlet.git $SCR/src
echo "code $(git -C $SCR/src rev-parse HEAD) task ${SLURM_ARRAY_TASK_ID:-0} node $(hostname)"
EDROP=$WORK/src/singlet-py/claude__empty-droplet-check-624070
OUT=$WORK/alpha_bench/compare; mkdir -p $OUT
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" --env "EDROP=$EDROP" \
  --env "CATALOG=$WORK/data/scbasecount/meta/catalog_map.parquet" --env "HDF5_USE_FILE_LOCKING=FALSE" \
  $WORK/images/python-3.12-bookworm.sif \
  python -P $SCR/src/bench/alpha/alpha_compare.py $SCR/src/bench/alpha/picks.tsv ${SLURM_ARRAY_TASK_ID:-0} $OUT
