#!/bin/bash
#SBATCH --job-name=hto-i2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# I2 (Stoeckius 2018 mixed cell lines). Pass TRUTH_ONLY=1 to build the RNA truth without running any caller.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/adt-bench/bin:$PATH
O=$W/expression-sota/hto/results; mkdir -p $O; cd $W/expression-sota/hto/code/bench/expression_sota/hto
export OMP_NUM_THREADS=4
EXTRA=""; [ "${TRUTH_ONLY:-0}" = "1" ] && EXTRA="--truth-only"
python -W ignore -P eval_independent.py --data $W/expression-sota/hto/data/stoeckius2018 --frozen frozen.json --only I2 --out $O/independent_I2.json $EXTRA
