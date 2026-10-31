#!/bin/bash
# The `kernels` stage (decode ceiling of textbook integer codes against .s1, one pinned core), every matrix in one job so
# every code is timed on the same node, one matrix after another:
#   sbatch --export=ALL,S1_SHA=<sha> kernels.sbatch          (S1_FORCE=1 redoes matrices whose results exist)
# The canonical matrices and .s1 files are read from the project space; only the timed decode is in memory.
#SBATCH --job-name=s1-kernels
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%j.out
set -euo pipefail
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
s1_activate
(
  flock 9
  [ -x "$BENCH/cpp/kernel_bench" ] || CXX=g++ bash "$BENCH/cpp/build.sh"
) 9>"$S1/src/.lock-cpp-$SHA"
echo "node $(hostname), $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
t0=$SECONDS
s1b kernels --datasets "${DS_LIST:-all}" ${S1_FORCE:+--force}
echo "kernels done in $((SECONDS - t0)) s"
