#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --time=00:10:00
#SBATCH --mem=2G
#SBATCH --job-name=cpuperf-sumdir
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/jobs/cpu-perf/sumdir.out
# D=<results dir> M=<reads in millions>: memory timeline, stages, threads for one PACK=1 run.
R=/mnt/projects/debruinz_project/debruinz/singlet/jobs/cpu-perf
python3 $R/mem.py $D; python3 $R/stage_profile.py $D $M | head -12; python3 $R/threads.py $D $M | head -4
