#!/bin/bash
#SBATCH --job-name=lr-truth
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# STEPS="b a final" (default) ; smoke test: OUT=... LR_RG=3 LR_REC=300000 sbatch --time=00:30:00 --partition=debug-cpu run_truth.sbatch
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/longread_usa
PY=/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python
OUT=${OUT:-$X/truth}; mkdir -p $OUT
for s in ${STEPS:-b a final}; do echo "== $s $(date -Is)"; /usr/bin/time -v -o $OUT/time_$s.txt $PY $X/code/truth_usa.py $s $OUT --workers 16; done
