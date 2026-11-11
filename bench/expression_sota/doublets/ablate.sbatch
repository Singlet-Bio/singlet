#!/bin/bash
#SBATCH -J dbl-ablate
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/ambient/bin:$PATH
python -I $W/expression-sota/code/doublets/ablate.py $W/expression-sota/doublets_ablate.csv
