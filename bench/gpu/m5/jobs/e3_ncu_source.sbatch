#!/bin/bash
#SBATCH --job-name=e3-src
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E3: per-source-line metrics (CUDA source view) of Nsight Compute reports made by e3_run.sbatch (NCUS): REP="dir/ncu_wm dir/ncu_bm"
# -> <report>_source.csv next to each report (metrics and source text of bench/gpu/m5 only; no read data)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; NCU=/usr/local/cuda/bin/ncu
for r in $REP; do
  apptainer exec -B $W $IMG $NCU --import $r.ncu-rep --page source --print-source cuda --csv > ${r}_source.csv 2> ${r}_source.err
  echo "$r: exit=$? lines=$(wc -l < ${r}_source.csv)"; head -c 600 ${r}_source.csv; echo; head -3 ${r}_source.err
done
