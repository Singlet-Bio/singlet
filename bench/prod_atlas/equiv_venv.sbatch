#!/bin/bash
#SBATCH --job-name=ar-equivvenv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# called_v2 equivalence (bench/called_v2/equiv_cpp.py) in the pinned runtime venv (python 3.12 image + venvs/singlet:
# numpy 2.2 / scipy 1.15 / numba 0.64), for the singlet-prod-t4 runs of validate.sbatch.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; SRC=$W/src/singlet-py/prod__atlas-rerun
RUNS=""
for ID in pbmc1k_v3 pbmc5p_v1 nuclei2k_v2 neuron1k_v3 hgmm1k_v3; do
  [ -f $PB/runs/$ID/singlet-prod-t4/cell_calls.tsv ] && RUNS="$RUNS $PB/runs/$ID/singlet-prod-t4"
done
apptainer exec -B /mnt/projects $W/images/python-3.12-bookworm.sif bash -c \
  "source $W/venvs/singlet/bin/activate && python -c 'import numpy, scipy, numba; print(\"venv: numpy\", numpy.__version__, \"scipy\", scipy.__version__, \"numba\", numba.__version__)' && cd $SRC && python -P bench/called_v2/equiv_cpp.py $RUNS --json $PB/results/prod_atlas/equiv_venv.json" 2>&1 | grep -v "Warning\|^ \|sp\.\|migration"
