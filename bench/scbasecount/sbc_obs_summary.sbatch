#!/bin/bash
#SBATCH --job-name=sbc-obs
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
D=$WORK/jobs/sbc
EDROP=$WORK/src/singlet-py/claude__empty-droplet-check-624070
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" --env "EDROP=$EDROP" \
  $WORK/images/python-3.12-bookworm.sif \
  python -P $D/sbc_obs_summary.py $WORK/data/scbasecount/coverage/sbc_obs_summary.parquet Homo_sapiens Mus_musculus Danio_rerio Macaca_mulatta Rattus_norvegicus
