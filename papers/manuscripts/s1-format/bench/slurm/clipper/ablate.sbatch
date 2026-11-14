#!/bin/bash
# The `ablate` stage (which parts of the .s1 model earn their bits), one dataset per array task or a list in DS_LIST:
#   sbatch --export=ALL,S1_SHA=<sha>,S1_OVERLAY=<dir>,DS_LIST=baron_pancreas ablate.sbatch          (one dataset)
#   sbatch --array=0-28%4 --export=ALL,S1_SHA=<sha>,S1_OVERLAY=<dir> ablate.sbatch                  (every dataset)
# S1_SHA is the commit whose codec and canonical matrices the paper used. Until the stage is merged, S1_OVERLAY holds its files
# (papers/manuscripts/s1-format/bench/...), which are copied over a private clone of that commit; leave S1_OVERLAY unset once the
# commit contains the stage. The canonical matrices are read from the project space; results go to $S1/results/ablate.
#SBATCH --job-name=s1-ablate
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%A_%a.out
set -euo pipefail
SRC0=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}
# the commit's own clone, if no earlier job made it (s1_checkout in common.sh does the same under a lock)
if [ ! -d "$SRC0" ]; then
  git clone -q https://github.com/Singlet-Bio/singlet.git "$SRC0.a$SLURM_JOB_ID" && git -C "$SRC0.a$SLURM_JOB_ID" checkout -q "$S1_SHA"
  mv -n "$SRC0.a$SLURM_JOB_ID" "$SRC0" 2>/dev/null || rm -rf "$SRC0.a$SLURM_JOB_ID"
fi
source "$SRC0/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
if [ -n "${S1_OVERLAY:-}" ]; then
  # a private clone, so the shared checkout other jobs use stays untouched
  AB=$S1/src/$SHA-ablate
  (
    flock 9
    if [ ! -d "$AB" ]; then git clone -q "$SRC" "$AB.part" && mv "$AB.part" "$AB"; fi
    cp -r "$S1_OVERLAY"/. "$AB"/
  ) 9>"$S1/src/.lock-ablate-$SHA"
  SRC=$AB
  BENCH=$SRC/papers/manuscripts/s1-format/bench
fi
s1_activate
(
  flock 9
  [ -x "$BENCH/cpp/ablate_bench" ] || CXX=g++ bash "$BENCH/cpp/build.sh"
) 9>"$S1/src/.lock-cpp-ablate-$SHA"
if [ -z "${DS_LIST:-}" ] && [ -n "${DS_ARRAY:-}" ]; then   # DS_ARRAY=a:b:c (colons: --export splits on commas): one task per listed dataset
  DS_LIST=$(echo "$DS_ARRAY" | cut -d: -f$((SLURM_ARRAY_TASK_ID + 1)))
fi
if [ -z "${DS_LIST:-}" ]; then
  DS_LIST=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['datasets'][int(sys.argv[2])]['id'])" \
            "$SRC/papers/manuscripts/s1-format/datasets.yaml" "${SLURM_ARRAY_TASK_ID:?set DS_LIST or run as an array}")
fi
echo "node $(hostname), $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2); datasets $DS_LIST"
t0=$SECONDS
s1b ${S1_ABLATE_MAX:+--set ablate.max_nnz=$S1_ABLATE_MAX} ablate --datasets "$DS_LIST" ${S1_FORCE:+--force}
echo "ablate done in $((SECONDS - t0)) s"
