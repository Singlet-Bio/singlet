#!/bin/bash
# Run a python script of the bpcalls work on a CPU node (D-163): SCRIPT=<path under $WORK/bpcalls/src> ARGS="..."; the repo copy is pushed to $WORK/bpcalls/src
#SBATCH --job-name=bpcalls
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/bpcalls-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$WORK/bpcalls/src
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" --env "PYTHONPATH=$SRC/python" \
  --env "OMP_NUM_THREADS=16" --env "OPENBLAS_NUM_THREADS=16" --env "PYTHONWARNINGS=ignore" --env "LQC=$WORK/latent_qc" \
  "$WORK/images/python-3.12-bookworm.sif" python -u "$SRC/$SCRIPT" $ARGS
