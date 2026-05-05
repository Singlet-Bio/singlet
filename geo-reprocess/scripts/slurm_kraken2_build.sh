#!/bin/bash
#SBATCH --job-name=kraken2_build
#SBATCH --partition=all
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kraken2_build_%A_%a.out
#SBATCH --error=logs/kraken2_build_%A_%a.err

# Build per-GSE kraken2.1pz matrices from per-GSM kraken2_cell_taxa.parquet
# Runs after merge_gse.py has completed all GSEs.

set -eo pipefail

module load python/3.9.23

cd /mnt/home/debruinz/Singlet-AI
export PYTHONPATH="/mnt/home/debruinz/Singlet-AI/singlepress:${PYTHONPATH:-}"

if [ -d ".bench_venv" ]; then
    source .bench_venv/bin/activate
fi

python scripts/build_kraken2.py \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks "${SLURM_ARRAY_TASK_COUNT}" \
    > "logs/kraken2_report_${SLURM_ARRAY_TASK_ID}.json"
