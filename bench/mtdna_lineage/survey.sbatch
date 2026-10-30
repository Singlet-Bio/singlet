#!/bin/bash
#SBATCH --job-name=mtl-survey
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# chrM reads per called cell over a random sample of human atlas samples (atlas_survey.py; reads data.singlet.bio).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/mtl; O=$W/out/mtdna_lineage/atlas; mkdir -p $O
$W/envs/mtl/bin/python $J/atlas_survey.py $J/sample_meta.csv $O/atlas_survey.csv --n-studies 400 --per-study 2 --seed 1
wc -l $O/atlas_survey.csv
