#!/bin/bash
#SBATCH --job-name=dbl-cheap
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
mkdir -p $WORK/latent_qc/dbl/cheap
in_runtime python $WORK/jobs/dbl-cheap/cheap_features.py $WORK/latent_qc/dbl/cheap/cheap_features.csv
