#!/bin/bash
#SBATCH --job-name=adt-run
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# One ADT/HTO library per array task. Submit with --array=0-N%K and LIST=<file of GSMs, one per line, under
# $WORK/state/adt/lists/>. Outputs: $WORK/data/adt/<GSE>/<GSM>/ (counts, panel, QC, status.json). FASTQs stay in /tmp.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/adt
BRANCH=claude/adt-modality-0637a6
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
: "${LIST:?set LIST}"
INV=${INV:-$WORK/state/adt/adt_hto_inventory.tsv}
SCRATCH=/tmp/$USER-adt-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v python -P $SRC/bench/adt/run_library.py --inventory $INV --gsm @$LIST --index $SLURM_ARRAY_TASK_ID \
  --out ${OUT:-$WORK/data/adt} --scratch $SCRATCH --cache $WORK/cache/adt --geo-panels $WORK/state/adt/panels_geo --threads $SLURM_CPUS_PER_TASK 2>&1 \
  | grep -vE "^\s*$|Page size|Exit status: 0"
