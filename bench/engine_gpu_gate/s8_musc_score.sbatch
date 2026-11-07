#!/bin/bash
#SBATCH --job-name=s8-muscscore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: score the U calls of the 389 and 397 runs against the MuSC long-read truth (s8_musc_score.py).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PY=/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python; O=$W/s8/musc
$PY -P $O/code/s8_musc_score.py $O/truth_lr.tsv.gz $O/score_musc.json r389=$O/singlet_389/molecules.tsv.gz r397=$O/singlet_397/molecules.tsv.gz
