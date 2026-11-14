#!/bin/bash
#SBATCH --job-name=cc-cbscore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/bench/calls-sota/cell_calling
$W/envs/gpool/venv/bin/python -P $S/cb_score.py $S/cb_scores.json
MODE=score $W/envs/gpool/venv/bin/python -P $S/nuc_fair.py score $S/nuc_fair $S/cellbender/nucleus_hash/cb_cell_barcodes.csv $S/nuc_fair/scores_with_cellbender.json
