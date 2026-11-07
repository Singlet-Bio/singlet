#!/bin/bash
#SBATCH --job-name=bp-ap-detsj
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
$PY -I $SRC/bench/public/harness/alignment_performance/determinism_sj.py $BP/determinism/t4_cache $BP/determinism/t8_cache 2>&1 | tee $BP/determinism/sj_compare.txt
