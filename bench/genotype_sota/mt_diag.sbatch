#!/bin/bash
#SBATCH --job-name=gsota-mtdiag
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
A=$W/bench-public/artifacts/genotype-sota-f15/mt; mkdir -p $A
python -I $W/jobs/gsota-r12/mt_diag.py $W/bench-public/runs $W/bench-public/data/kang_truth $A/mt_diag_ctrl.tsv
