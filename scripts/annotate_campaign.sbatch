#!/bin/bash
# Atlas-wide per-cell annotation on Clipper: one array task = one shard of studies (scripts/annotate_campaign.py).
#
# Prepare once (a job, not the login node):
#   python scripts/annotate_campaign.py plan --studies studies.tsv --shards 400 --out $OUT/plan.tsv
# Submit (ASK ZACH FIRST: this is the atlas-wide run), throttled to 60 concurrent tasks:
#   sbatch --array=0-399%60 --export=ALL,OUT=$OUT,AF=$AF,SRC=$SRC scripts/annotate_campaign.sbatch
# Status:
#   python scripts/annotate_campaign.py status --out $OUT
#
# Sizing (validation run, docs/cell_annotation.md): the time is in reading counts and the per-sample self-training; about
# 400 CPU-hours for the atlas at 8 threads per task. A shard of ~10 GB of bundles fits a 4 h short-cpu slot.
#SBATCH --job-name=annotate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
: "${SRC:?set SRC to a synced worktree of Singlet-Bio/singlet}" "${OUT:?set OUT}"
AF=${AF:-$WORK/data/annotate/1kgp_superpop_af.parquet}
source $WORK/src/singlet-agents/main/hpc/clipper/in_runtime.sh
SCRATCH=/tmp/$USER-annotate-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT   # bundles and restricted sidecars live only here
# R2 read credentials for the restricted sidecars: exported from a 600-mode file, inherited by the container through the
# environment (never on a command line, where other users' `ps` would see them); without them ancestry stays null
[ -f "$HOME/.config/singlet-r2.env" ] && source "$HOME/.config/singlet-r2.env"
export PYTHONPATH=$SRC/python SINGLET_CACHE_DIR=$SCRATCH/cache OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
in_runtime python -u $SRC/scripts/annotate_campaign.py run --plan $OUT/plan.tsv --shard $SLURM_ARRAY_TASK_ID --out $OUT --af $AF \
  --scratch $SCRATCH --ledger $OUT/ledger/shard-$SLURM_ARRAY_TASK_ID.jsonl
