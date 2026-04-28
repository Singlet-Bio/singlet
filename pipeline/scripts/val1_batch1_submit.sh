#!/bin/bash
# val1_batch1_submit.sh — Submit VAL1 batch 1 (first 50 samples, lines 2-51)
# Run from: /mnt/home/debruinz/Singlet-AI/singlify/
#
# Dependency chain: download → process
# (val1_download.sh produces .1fq directly; no separate encode step needed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../val1_logs"
mkdir -p "$LOG_DIR"

echo "Submitting VAL1 batch 1 (array lines 2-51, 50 samples)..."

# Download (+ encode): singlify download → .1fq
DL_JOB=$(sbatch --array=2-51%10 "${SCRIPT_DIR}/val1_download.sh" | awk '{print $4}')
echo "Download job: ${DL_JOB}"

# Process: depends on all downloads completing
PROC_JOB=$(sbatch --dependency=afterok:"${DL_JOB}" --array=2-51%5 "${SCRIPT_DIR}/val1_process.sh" | awk '{print $4}')
echo "Process job:  ${PROC_JOB} (depends on ${DL_JOB})"

echo ""
echo "Monitor with: squeue -j ${DL_JOB},${PROC_JOB}"
echo "Logs in:      ${LOG_DIR}/"
