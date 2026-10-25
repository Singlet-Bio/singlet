#!/bin/bash
#SBATCH --job-name=adt-pub
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Public ADT benchmark (capability adt): one cohort dataset per array task (0 pbmc1k_cite_v3, 1 pbmc10k_cite_v3, 2 pbmc5p_v1).
# Every tool runs on the same antibody FASTQs with 8 threads; artifacts -> $WORK/bench-public/artifacts/adt/
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$WORK/src/singlet-py/claude__adt-modality-0637a6
DS=(pbmc1k_cite_v3 pbmc10k_cite_v3 pbmc5p_v1); D=${DS[$SLURM_ARRAY_TASK_ID]}
OUT=$WORK/bench-public/artifacts/adt; mkdir -p $OUT
SCRATCH=/tmp/$USER-adtpub-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
export PATH=$WORK/envs/adt/bin:$PATH:$WORK/envs/adt-csc/bin LD_LIBRARY_PATH=$WORK/envs/adt/lib
export AFRY_BIN=$WORK/envs/adt-afry/bin PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
R="python -P $SRC/bench/adt/adt_public_run.py --dataset $D --data $WORK/data/adt_bench --out $OUT --scratch $SCRATCH --threads 8"
$R --tool prepare
for t in ${TOOLS:-singlet kite citeseqcount alevinfry}; do $R --tool $t || echo "tool $t failed"; done
