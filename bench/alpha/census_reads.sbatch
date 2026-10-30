#!/bin/bash
#SBATCH --job-name=alpha-census
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git clone -q --depth 1 -b ${1:-claude/alpha-bench} git@github.com:Singlet-Bio/singlet.git $SCR/src
mkdir -p $WORK/alpha_bench/census
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
  $WORK/images/python-3.12-bookworm.sif python -P $SCR/src/bench/alpha/census_reads.py \
  $WORK/alpha_bench/census/alpha_public_gsms.tsv $WORK/pipeline-benchmark/results/campaign/campaign_samples.csv \
  $WORK/data/scbasecount/meta/catalog_map.parquet $WORK/alpha_bench/census/alpha_reads.tsv.gz
