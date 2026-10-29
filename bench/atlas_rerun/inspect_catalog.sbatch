#!/bin/bash
#SBATCH --job-name=ar-inspect
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu,scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python -I $W/jobs/atlas-rerun/inspect_catalog.py $W/state/v2/plan-09fd92b149/plan.tsv
ls -la /mnt/projects/debruinz_project/cellarium/catalog/ | head -30
