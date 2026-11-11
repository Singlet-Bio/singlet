#!/bin/bash
#SBATCH --job-name=lr-score2
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Re-score the long-read truth with Singlet's new molecule labels (run_singlet_labels.sbatch); competitors and truth are the earlier runs.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/longread_usa; X2=$X/rescore_usa_labels
for d in truth starsolo alevinfry; do [ -e $X2/$d ] || ln -s $X/$d $X2/$d; done
/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python $X/code/score_usa.py $X2 $X2/score_usa.json
