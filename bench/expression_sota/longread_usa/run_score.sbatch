#!/bin/bash
#SBATCH --job-name=lr-score
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/longread_usa
/usr/bin/time -v -o $X/time_score.txt /mnt/projects/debruinz_project/debruinz/envs/gt/bin/python $X/code/score_usa.py $X $X/score_usa.json
