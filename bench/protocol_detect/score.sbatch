#!/bin/bash
#SBATCH --job-name=detect-score
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Re-scores a finished panel directory (DIR) with score_panel.py.
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/bpub/bin/python3 $W/jobs/detect/src/score_panel.py ${DIR:?} > $DIR/score.txt; cat $DIR/score.txt
