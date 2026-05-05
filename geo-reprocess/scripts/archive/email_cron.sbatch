#!/bin/bash
#SBATCH --job-name=email-rpt
#SBATCH --partition=cpu
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/home/debruinz/logs/status_email_%j.out
#SBATCH --error=/mnt/home/debruinz/logs/status_email_%j.err
#
# Self-resubmitting SLURM job that sends a pipeline status email every 6 hours.
#
# To start:  sbatch geo-reprocess/scripts/email_cron.sbatch
# To stop:   scancel -n email-rpt

set -euo pipefail

SCRIPT_DIR="/mnt/home/debruinz/Singlet-AI/geo-reprocess/scripts"

echo "[$(date -u +%FT%TZ)] Sending status email..."
cd /mnt/home/debruinz/Singlet-AI
python3 "$SCRIPT_DIR/send_status_email.py"
echo "[$(date -u +%FT%TZ)] Done."

# Resubmit self to run again in 6 hours
sbatch --begin="now+6hours" "$SCRIPT_DIR/email_cron.sbatch"
echo "[$(date -u +%FT%TZ)] Resubmitted for 6 hours from now."
