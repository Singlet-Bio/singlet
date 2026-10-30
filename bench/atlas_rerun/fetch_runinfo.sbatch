#!/bin/bash
#SBATCH --job-name=ar-runinfo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# ENA run report for every project behind the v2 plan studies (see fetch_runinfo.py).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python -I $W/jobs/atlas-rerun/fetch_runinfo.py $W/state/v2/plan-09fd92b149/plan.tsv $W/data/atlas-rerun
