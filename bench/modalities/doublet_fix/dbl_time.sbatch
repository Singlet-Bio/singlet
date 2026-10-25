#!/bin/bash
#SBATCH --job-name=dbl-time
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Wall time of detect_doublets() (defaults, exon + intron sources with maps, as export.h) on the Kang cells and on 3x / 4x column copies
# (43k / 58k cells; copies make exact ties but cost the same).  Uses all 16 threads.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/modfix-doublet; O=$W/modality-bench/doublet_fix
export PATH=$W/envs/pbench/bin:$PATH
mkdir -p $O/build $O/results $O/time
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$J/inc $J/dbl_run.cpp -o $O/build/dbl_run_time
: > $O/results/timing.jsonl
for T in 1 3 4; do
  /usr/bin/time -f "wall_s=%e maxrss_kb=%M" $O/build/dbl_run_time $O/kang/k8 $O/time/t$T.tsv threads=16 second=$O/kang/k8_in tile=$T ${ARGS:-} 2> $O/time/t$T.log
  echo "tile=$T $(cat $O/time/t$T.log | tr '\n' ' ')" | tee -a $O/results/timing.jsonl
done
