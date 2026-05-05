#!/bin/bash
#SBATCH --job-name=metrics
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/home/debruinz/logs/dashboard_metrics_%j.out
#SBATCH --error=/mnt/home/debruinz/logs/dashboard_metrics_%j.err
#
# Self-resubmitting SLURM job that collects pipeline metrics hourly
# and pushes to the website repo.
#
# To start:  sbatch geo-reprocess/scripts/metrics_cron.sbatch
# To stop:   scancel -n metrics

set -euo pipefail

SCRIPT_DIR="/mnt/home/debruinz/Singlet-AI/geo-reprocess/scripts"
COLLECTOR="$SCRIPT_DIR/collect_dashboard_metrics.py"

echo "[$(date -u +%FT%TZ)] Running metrics collector..."
cd /mnt/home/debruinz/Singlet-AI
python3 "$COLLECTOR"
echo "[$(date -u +%FT%TZ)] Done."

# Resubmit self to run again in 1 hour
sbatch --begin="now+60minutes" "$SCRIPT_DIR/metrics_cron.sbatch"
echo "[$(date -u +%FT%TZ)] Resubmitted for next hour."
