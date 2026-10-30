#!/bin/bash
#SBATCH --job-name=amb-tools
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch tools.sbatch      (after singlet_run.sbatch)
# (after prep.sbatch) SoupX / DecontX / scDblFinder (R) and Scrublet (python) on Singlet's called cells.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/ambient; S=$W/jobs/modalities/ambient
export PATH=$E/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl NUMBA_CACHE_DIR=$W/cache/numba-edrop OMP_NUM_THREADS=8
mkdir -p $W/modality-bench/ambient/results
python -I $S/run_scrublet.py
Rscript $S/run_r_tools.R
ls -la $W/modality-bench/ambient/results
