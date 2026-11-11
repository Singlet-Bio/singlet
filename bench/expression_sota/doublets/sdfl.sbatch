#!/bin/bash
#SBATCH --job-name=sdfl
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sbatch sdfl.sbatch VARIANT   (a|b|c)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/code/doublets
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome OMP_NUM_THREADS=16 SDFL_VARIANT=${1:-a} PYTHONWARNINGS=ignore
NAMES=$(ls $W/latent_qc/dbl/data | grep -v -E "synth|homsim")
python -I $X/sdfl.py $NAMES
V=sdfl; [ "$SDFL_VARIANT" != a ] && V=sdfl$SDFL_VARIANT
python -I $X/eval_methods.py $W/expression-sota/doublets_$V.csv scdblfinder $V prod combo_prod_atlas combo_all
