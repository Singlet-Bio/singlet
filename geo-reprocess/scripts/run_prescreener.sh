#!/bin/bash
#SBATCH --job-name=prescreen-ss
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/prescreen_%j.out
#SBATCH --error=/mnt/projects/debruinz_project/cellarium/pipeline/logs/prescreen_%j.err

# Standalone pre-screener: FASTQ-peek all unknown_sc / snRNA_unknown samples
# and write skip manifests for confirmed smartseq. Safe to re-run (idempotent).
#
# Run: sbatch run_prescreener.sh

set -euo pipefail

module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
conda activate cellarium

cd /mnt/home/debruinz/Singlet-AI/geo-reprocess
python3 -u /mnt/home/debruinz/Singlet-AI/scripts/smartseq_prescreener.py

echo "Pre-screening complete."
