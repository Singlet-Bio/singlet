#!/bin/bash
#SBATCH --job-name=adt-htoz
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# HTO demultiplexing on the Howitt et al. 2023 benchmark (Zenodo 8304003) -> $WORK/state/adt/hto_zenodo/
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$WORK/src/singlet-py/claude__adt-modality-0637a6
Z=$WORK/data/adt_bench/zenodo_8304003
OUT=$WORK/state/adt/hto_zenodo
export PATH=$WORK/envs/adt-bench/bin:$PATH LD_LIBRARY_PATH=$WORK/envs/adt-bench/lib:$WORK/envs/adt/lib
export PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome OMP_NUM_THREADS=4
[ -f $OUT/published/batch1_published_calls.csv ] || Rscript $SRC/bench/adt/export_rds_calls.R $Z $OUT/published
python -P $SRC/bench/adt/hto_zenodo.py --zenodo $Z --published $OUT/published --out $OUT ${METHODS:+--methods $METHODS}
