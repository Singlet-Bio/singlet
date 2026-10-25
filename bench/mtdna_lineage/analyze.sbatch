#!/bin/bash
#SBATCH --job-name=mtl-analyze
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch analyze.sbatch <lines|bpdcn|size>
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/mtl; R=$W/out/mtdna_lineage; mkdir -p $J/results
export OMP_NUM_THREADS=8 MPLCONFIGDIR=$W/cache/mpl
/usr/bin/time -v -o $R/analyze_$1.time.txt $W/envs/mtl/bin/python $J/analyze.py $1 $R $J/results/$1.json
grep -E "Elapsed|Maximum res" $R/analyze_$1.time.txt
