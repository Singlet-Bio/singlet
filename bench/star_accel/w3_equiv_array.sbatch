#!/bin/bash
#SBATCH --job-name=w3-equiv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch --array=0-N%4 w3_equiv_array.sbatch <manifest.tsv>
# One w3_equiv.sbatch comparison per manifest line (tab-separated: dataset binA binB tag [ENV_A] [ENV_B]; '-' = none),
# line index = SLURM_ARRAY_TASK_ID (comment lines starting with # are skipped).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet
LINE=$(grep -v '^#' "$1" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
IFS=$'\t' read -r DS A B TAG EA EB <<< "$LINE"
[ "${EA:--}" = "-" ] && EA=""; [ "${EB:--}" = "-" ] && EB=""
ENV_A="$EA" ENV_B="$EB" bash $W/src/singlet-py/perf__w3-speedups/bench/star_accel/w3_equiv.sbatch "$DS" "$A" "$B" "$TAG"
