#!/bin/bash
# Untimed per-dataset stages, one dataset per array task. Choose the partition and memory per size class at submission:
#   sbatch --array=0-N%K --partition=... --mem=... --export=ALL,S1_SHA=<sha>,DS_LIST=a,b,c,STAGES="fetch canon" stage.sbatch
# DS_LIST: comma-separated dataset ids (array index picks one). STAGES: s1bench stages, run in order.
# EXT=1 adds --extensions (needed for the out-of-domain datasets).
#SBATCH --job-name=s1-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%A_%a.out
set -euo pipefail
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
s1_activate
IFS=, read -ra DS <<<"${DS_LIST:?set DS_LIST}"
D=${DS[${SLURM_ARRAY_TASK_ID:-0}]}
echo "dataset $D stages ${STAGES:?set STAGES}"
for st in $STAGES; do
  t0=$SECONDS
  s1b "$st" --datasets "$D" ${EXT:+--extensions}
  echo "stage $st $D done in $((SECONDS - t0)) s"
done
