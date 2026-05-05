#!/bin/bash
# Setup script for the hourly pipeline metrics collector cron job.
#
# This installs a cron job that:
#   1. Runs collect_dashboard_metrics.py every hour at :05
#   2. Writes pipeline-metrics.json to the singletai-website repo
#   3. Commits and pushes to GitHub automatically
#
# Usage:
#   bash geo-reprocess/scripts/setup_metrics_cron.sh        # Install
#   bash geo-reprocess/scripts/setup_metrics_cron.sh remove  # Remove

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLLECTOR="$SCRIPT_DIR/collect_dashboard_metrics.py"
LOGFILE="/mnt/home/debruinz/logs/dashboard_metrics.log"
CRON_TAG="# singlet-dashboard-metrics"

JOB="5 * * * * cd /mnt/home/debruinz/Singlet-AI && /usr/bin/python3 $COLLECTOR >> $LOGFILE 2>&1 $CRON_TAG"

if [[ "${1:-}" == "remove" ]]; then
    crontab -l 2>/dev/null | grep -v "$CRON_TAG" | crontab -
    echo "Removed dashboard metrics cron job."
    exit 0
fi

# Ensure log directory exists
mkdir -p "$(dirname "$LOGFILE")"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "$CRON_TAG"; then
    echo "Cron job already installed. Use 'remove' to uninstall first."
    crontab -l | grep "$CRON_TAG"
    exit 0
fi

# Install
(crontab -l 2>/dev/null; echo "$JOB") | crontab -
echo "Installed hourly cron job:"
echo "  $JOB"
echo ""
echo "Logs at: $LOGFILE"
echo "To test: python3 $COLLECTOR --dry-run"
echo "To remove: bash $0 remove"
