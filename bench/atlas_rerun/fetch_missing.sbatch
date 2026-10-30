#!/bin/bash
#SBATCH --job-name=ar-missing
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
cd $W/jobs/atlas-rerun
in_runtime python fetch_missing.py ${1:-$W/data/atlas-rerun/wanted.tsv} ${2:-$W/data/atlas-rerun/runs_extra.parquet}
