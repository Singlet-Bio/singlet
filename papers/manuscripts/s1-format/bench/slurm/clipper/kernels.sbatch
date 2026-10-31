#!/bin/bash
# The `kernels` stage (decode ceiling of textbook integer codes against .s1, one pinned core), every matrix in one job so
# every code is timed on the same node, one matrix after another:
#   sbatch --export=ALL,S1_SHA=<sha> kernels.sbatch          (S1_FORCE=1 redoes matrices whose results exist;
#                                                            S1_STAGE=entropy runs the untimed entropy stage instead)
# The canonical matrices and .s1 files are read from the project space; only the timed decode is in memory.
#SBATCH --job-name=s1-kernels
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%j.out
set -euo pipefail
# the commit's own clone, if no earlier job made it (s1_checkout in common.sh does the same under a lock)
SRC0=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}
if [ ! -d "$SRC0" ]; then
  git clone -q https://github.com/Singlet-Bio/singlet.git "$SRC0.k$SLURM_JOB_ID" && git -C "$SRC0.k$SLURM_JOB_ID" checkout -q "$S1_SHA"
  mv -n "$SRC0.k$SLURM_JOB_ID" "$SRC0" 2>/dev/null || rm -rf "$SRC0.k$SLURM_JOB_ID"
fi
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
s1_activate
(
  flock 9
  [ -x "$BENCH/cpp/kernel_bench" ] || CXX=g++ bash "$BENCH/cpp/build.sh"
) 9>"$S1/src/.lock-cpp-$SHA"
echo "node $(hostname), $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
t0=$SECONDS
s1b "${S1_STAGE:-kernels}" --datasets "${DS_LIST:-all}" ${S1_FORCE:+--force}
echo "${S1_STAGE:-kernels} done in $((SECONDS - t0)) s"
