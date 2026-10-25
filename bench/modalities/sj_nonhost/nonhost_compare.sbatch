#!/bin/bash
#SBATCH --job-name=sjnh-nhcmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; B=$W/modality-bench/sj_nonhost
export PBMC_SINGLET=$B/singlet/pbmc1k_v3
$W/envs/pbench/bin/python -I -W ignore $B/scripts/nonhost_compare.py $B $B/nonhost_compare.json
