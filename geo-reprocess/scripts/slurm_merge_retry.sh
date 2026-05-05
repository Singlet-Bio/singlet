#!/bin/bash
#SBATCH --job-name=merge_retry
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_retry_%A_%a.out
#SBATCH --error=logs/merge_retry_%A_%a.err

# Retry merge for GSEs that OOM'd. Uses --gse-list with array indexing.
# Memory and array size set at submission time via --mem and --array.
#
# Usage:
#   sbatch --mem=48G --array=0-261 scripts/slurm_merge_retry.sh logs/retry_medium.txt
#   sbatch --mem=128G --array=0-12 --partition=bigmem scripts/slurm_merge_retry.sh logs/retry_big.txt

set -eo pipefail

GSE_LIST="${1:?Usage: sbatch slurm_merge_retry.sh <gse_list_file>}"

module load python/3.9.23

cd /mnt/home/debruinz/Singlet-AI
export PYTHONPATH="/mnt/home/debruinz/Singlet-AI/singlepress:${PYTHONPATH:-}"

if [ -d ".bench_venv" ]; then
    source .bench_venv/bin/activate
fi

# Each array task processes one GSE from the list
GSE_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$GSE_LIST")

if [ -z "$GSE_ID" ]; then
    echo "No GSE for task $SLURM_ARRAY_TASK_ID" >&2
    exit 0
fi

echo "Processing $GSE_ID (task $SLURM_ARRAY_TASK_ID)" >&2
python scripts/merge_gse.py "$GSE_ID" 2>&1
