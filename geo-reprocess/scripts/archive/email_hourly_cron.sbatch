#!/bin/bash
#SBATCH --job-name=email-hourly
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/home/debruinz/logs/status_email_hourly_%j.out
#SBATCH --error=/mnt/home/debruinz/logs/status_email_hourly_%j.err
#
# Self-resubmitting SLURM job that sends detailed pipeline status email every hour.
# Runs for 6 iterations (6 hours), then stops. After that, the normal 6-hour
# email_cron.sbatch takes over again.
#
# To start:  sbatch geo-reprocess/scripts/email_hourly_cron.sbatch
# To stop:   scancel -n email-hourly

set -euo pipefail

SCRIPT_DIR="/mnt/home/debruinz/Singlet-AI/geo-reprocess/scripts"
COUNTER_FILE="/tmp/email_hourly_counter"

# Read and increment counter
if [[ -f "$COUNTER_FILE" ]]; then
    COUNT=$(cat "$COUNTER_FILE")
else
    COUNT=0
fi
COUNT=$((COUNT + 1))
echo "$COUNT" > "$COUNTER_FILE"

echo "[$(date -u +%FT%TZ)] Hourly email iteration $COUNT/6"

# First refresh the metrics JSON so the email has fresh data
echo "[$(date -u +%FT%TZ)] Refreshing pipeline metrics..."
cd /mnt/home/debruinz/Singlet-AI
python3 "$SCRIPT_DIR/collect_dashboard_metrics.py" 2>&1 || echo "Warning: metrics collection had errors"

# Send detailed email
echo "[$(date -u +%FT%TZ)] Sending detailed status email..."
python3 "$SCRIPT_DIR/send_detailed_status_email.py"
echo "[$(date -u +%FT%TZ)] Done."

if [[ $COUNT -lt 6 ]]; then
    # Resubmit for next hour
    sbatch --begin="now+1hour" "$SCRIPT_DIR/email_hourly_cron.sbatch"
    echo "[$(date -u +%FT%TZ)] Resubmitted for 1 hour from now (iteration $((COUNT+1))/6)."
else
    # Done with hourly cadence, restart the normal 6-hour schedule
    echo "[$(date -u +%FT%TZ)] Hourly burst complete (6/6). Restarting 6-hour schedule."
    rm -f "$COUNTER_FILE"
    sbatch --begin="now+6hours" "$SCRIPT_DIR/email_cron.sbatch"
    echo "[$(date -u +%FT%TZ)] Resubmitted 6-hour email cron."
fi
