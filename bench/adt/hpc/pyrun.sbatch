#!/bin/bash
#SBATCH --job-name=adt-py
#SBATCH --account=debruinz
#SBATCH --partition=debug-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Run a small Python script ($PY, a path on Clipper) in the ADT env with this branch on the path (checks and reports).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/adt
SRC=$WORK/src/singlet-py/claude__adt-modality-0637a6
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$SRC/python:$SRC/bench/adt HOME=$WORK/cache/condahome
python -P $PY ${ARGS:-}
