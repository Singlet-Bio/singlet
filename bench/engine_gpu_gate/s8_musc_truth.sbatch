#!/bin/bash
#SBATCH --job-name=s8-musctruth
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: long-read USA truth of the MuSC cells (s8_musc_truth.py), written under $W/s8/musc/.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PY=/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python
O=$W/s8/musc; mkdir -p $O
$PY -P $W/s8/musc/code/s8_musc_truth.py $O/truth_lr.tsv.gz --workers 4
ls -l $O
