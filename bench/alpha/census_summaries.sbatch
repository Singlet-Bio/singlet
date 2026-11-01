#!/bin/bash
#SBATCH --job-name=alpha-summ
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# summary.json of every Alpha public sample, 8 shards of studies -> $WORK/alpha_bench/census/summaries_<k>.tsv
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git clone -q --depth 1 -b ${1:-claude/alpha-bench} git@github.com:Singlet-Bio/singlet.git $SCR/src
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
  $WORK/images/python-3.12-bookworm.sif python -P $SCR/src/bench/alpha/census_summaries.py \
  $WORK/alpha_bench/census/alpha_public_gsms.tsv $WORK/alpha_bench/census/summaries_$SLURM_ARRAY_TASK_ID.tsv \
  --shard $SLURM_ARRAY_TASK_ID --of 8 --threads 16
