#!/bin/bash
#SBATCH --job-name=adt-harvest
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Authors' feature references from GEO (small text files only) -> $WORK/state/adt/panels_geo/
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/adt
SRC=$WORK/src/singlet-py/claude__adt-modality-0637a6
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
python -P $SRC/bench/adt/harvest_panels.py --inventory $WORK/state/adt/adt_hto_inventory.tsv --out $WORK/state/adt/panels_geo
