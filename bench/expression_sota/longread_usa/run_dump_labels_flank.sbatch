#!/bin/bash
#SBATCH --job-name=lr-dumpanfl
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
X=/mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa
/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python $X/code/dump_labels.py $X $X/labels_dump_flank/dump_labels.json labels_dump_flank
