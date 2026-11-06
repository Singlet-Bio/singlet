#!/bin/bash
#SBATCH --job-name=dmx-summary
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth
$W/envs/gpool/venv/bin/python $J/repo/bench/demux-truth/summarize.py ${RESULTS:-$J/results/overnight}
