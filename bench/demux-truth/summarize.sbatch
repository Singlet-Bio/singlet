#!/bin/bash
#SBATCH --job-name=dmx2-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Scores every mixture and writes $J/results/summary.md (aggregate metrics only; genotypes stay in $J/private).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth2
$W/envs/gpool/venv/bin/python $J/repo/bench/demux-truth/summarize.py $J/results | tail -5
ls -la $J/results/summary.md
