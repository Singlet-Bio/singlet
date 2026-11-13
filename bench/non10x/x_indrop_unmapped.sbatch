#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --job-name=vn-indunm
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
J=$W/jobs/validation-engine/non10x; R=$W/results/validation-engine/non10x/ab2/indrop_500k/res
python3 -I $J/indrop_unmapped.py $R/star_Unmapped.out.mate1 500000
