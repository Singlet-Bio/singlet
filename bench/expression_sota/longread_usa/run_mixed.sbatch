#!/bin/bash
#SBATCH --job-name=lr-mixed
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
X=/mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa
/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python $X/code/mixed_check.py $X
