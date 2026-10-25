#!/bin/bash
#SBATCH --job-name=pb-costs
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
mkdir -p $PB/results/campaign
python -I $SRC_NEW/bench/pipeline/analysis/campaign_costs.py \
  /mnt/projects/debruinz_project/cellarium/full_campaign/results $PB/results/campaign/campaign_samples.csv
