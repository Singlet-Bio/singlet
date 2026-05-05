#!/bin/bash
#SBATCH --job-name=merge_gse
#SBATCH --partition=all
#SBATCH --array=0-499
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge_gse_%A_%a.out
#SBATCH --error=logs/merge_gse_%A_%a.err

# Merge per-GSM .1pz files into per-GSE consolidated counts.1pz
# Each array task processes ~7 GSEs (3,309 total / 500 tasks)
#
# Usage:
#   mkdir -p logs
#   sbatch slurm_merge_gse.sh
#
# Monitor:
#   squeue -u $USER -n merge_gse
#   tail -f logs/merge_gse_*_0.err

set -eo pipefail

module load python/3.9.23

cd /mnt/home/debruinz/Singlet-AI
export PYTHONPATH="/mnt/home/debruinz/Singlet-AI/singlepress:${PYTHONPATH:-}"

# Use shared venv if available for compiled singlepress extension
if [ -d ".bench_venv" ]; then
    source .bench_venv/bin/activate
fi

python scripts/merge_gse.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}" \
    > "logs/merge_gse_report_${SLURM_ARRAY_TASK_ID}.json"
