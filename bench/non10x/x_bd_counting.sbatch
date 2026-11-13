#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --job-name=vn-bdcount
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
python3 -I $W/jobs/validation-engine/non10x/bd_counting.py $W/jobs/non10x/out-v2/bd_wta_run1_4tags/res 2>&1 | grep -v -i warn
