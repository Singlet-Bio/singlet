#!/bin/bash
#SBATCH -J sdfl-kcl
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# coarse-cluster heterotypic pairs: sbatch kcl.sbatch KCL
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/code/doublets
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome OMP_NUM_THREADS=16 SDFL_KCL=$1 SDFL_TAG=_k$1 PYTHONWARNINGS=ignore
NAMES=$(ls $W/latent_qc/dbl/data | grep -v -E "synth|homsim")
python -I $X/sdfl.py $NAMES 2>&1 | grep -v -i warn | grep "^\["
python -I $X/eval_methods.py $W/expression-sota/doublets_k$1.csv scdblfinder sdfl_k$1 sdfl prod 2>&1 | tail -22
