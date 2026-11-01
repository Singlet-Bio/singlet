#!/bin/bash
# v2 mode of the doublets capability on one development dataset's gate-binary run (see v2_gate.py), then the contract document.
#   sbatch --export=ALL,SRC=<checkout of this branch>,DS=hgmm1k_v3,STUDY=10x_hgmm,RUN=<binary run dir>,TABLE=<ds>_doublets.tsv.gz,OUT=<artifact dir>[,TRUTH=text] bench/doublets/v2_gate.sbatch
#SBATCH --job-name=dbl-v2-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
source $WORK/src/singlet-agents/main/hpc/clipper/in_runtime.sh
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$SRC/python
mkdir -p "$OUT"
in_runtime python -u $SRC/bench/doublets/v2_gate.py --dataset $DS --study $STUDY --run $RUN --table $TABLE --out $OUT --threads $SLURM_CPUS_PER_TASK ${TRUTH:+--truth "$TRUTH"}
in_runtime bash -c "cd $SRC && python bench/public/run_capability.py doublets $OUT --out $OUT/doublets.json && python bench/public/contract.py validate $OUT/doublets.json --cohort bench/public/cohort.tsv"
