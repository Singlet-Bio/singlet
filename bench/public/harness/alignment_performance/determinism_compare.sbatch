#!/bin/bash
#SBATCH --job-name=bp-ap-detcmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
D=$BP/determinism
for pair in "t4_cache t8_cache" "t4_nocache t8_nocache" "t4_cache t4_nocache" "t8_cache t8_nocache"; do
  set -- $pair; echo "=================== $1 vs $2"
  $PY -I $SRC/bench/public/harness/alignment_performance/compare_dirs.py $D/$1 $D/$2
done 2>&1 | tee $D/compare.txt
