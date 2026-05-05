#!/usr/bin/env bash
#SBATCH --job-name=scgeo-qt-sra
#SBATCH --account=bio260157
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/anvil/projects/x-bio260157/scgeo/pipeline/logs/qt_sra_%j.out
#
# ── SRA-only Quantification phase (P4) ───────────────────────────────────
# Dedicated job for s3_sra_pending samples that need fasterq-dump + pigz.
# - QUANT_BATCH=1: one sample per job avoids serialized fasterq-dump blocking
# - 24h time limit: large samples (>600GB FASTQs) can take 6-10h
# - 32G RAM: fasterq-dump needs more memory for large SRR files
# - FASTQ_ONLY=false: process ONLY s3_sra_pending samples
# SU cost: max(32/4, 8) = 8 SU/hr
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="${PROJECT:-/anvil/projects/x-bio260157}"
SCRATCH="${SCRATCH:-/anvil/scratch/x-zdebruine}"

module load anaconda
conda activate "$PROJECT/envs/scgeo"
umask 0022

export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_INDEX_DIR="$SCGEO_BASE/index"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"
export SCGEO_PIPELINE_DIR="$SCGEO_BASE/pipeline"
export SCGEO_CATALOG_DIR="$SCGEO_BASE/catalog"
export PYTHONUNBUFFERED=1

export DL_DIR="$SCRATCH/scgeo_downloads"
export RESULT_DIR="$SCGEO_BASE/pipeline/results"
export QUANT_BATCH="${QUANT_BATCH:-1}"    # one sample per SRA job
export FASTQ_ONLY="false"                 # process ONLY s3_sra_pending samples

cd /tmp

mkdir -p "$RESULT_DIR"

echo "════════════════════════════════════════════════════"
echo "SRA Quant | job ${SLURM_JOB_ID:-local} | $(hostname) | $SLURM_CPUS_PER_TASK CPUs"
echo "Memory: $(free -g | awk '/Mem:/{print $2}')GB total"
echo "════════════════════════════════════════════════════"

# Run the main quant logic — env vars already set above override the defaults
bash "$PROJECT/geo-reprocess/scripts/anvil/anvil_quant.sh"

echo "SRA Quant phase complete."
