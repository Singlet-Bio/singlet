#!/bin/bash
#SBATCH --job-name=gbold-trim
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# trim the ncu SASS exports of job ${JOB} to the hot lines (CPU only)
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/gpu-bold/results/ncufull-${JOB:-800533}
for i in 1 2 3; do [ -f $R/ncu_source_$i.txt ] && $W/envs/pbench/bin/python3 $W/jobs/gpu-bold/jobs/ncu_source_trim.py $R/ncu_source_$i.txt > $R/ncu_source_trim_$i.txt 2>&1; ls -la $R/ncu_source_trim_$i.txt; done
