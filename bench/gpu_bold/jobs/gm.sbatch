#!/bin/bash
#SBATCH --job-name=gbold-gm
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# per-second device metrics of the nsys exports of job 800350 (CPU only)
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/gpu-bold/results/tp-${JOB:-800350}
for d in $R/*/c*; do [ -f $d/prof.sqlite ] || continue; echo "== $d"; $W/envs/pbench/bin/python3 $W/jobs/gpu-bold/jobs/gm_series.py $d/prof.sqlite $d/gm_series.csv > $d/gm_series.txt 2>&1; tail -9 $d/gm_series.txt; done
