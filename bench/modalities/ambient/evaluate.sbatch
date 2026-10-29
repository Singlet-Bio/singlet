#!/bin/bash
#SBATCH --job-name=amb-eval
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/ambient; S=$W/jobs/modalities/ambient
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
python -I $S/evaluate.py | tee $W/modality-bench/ambient/results/evaluate.out
