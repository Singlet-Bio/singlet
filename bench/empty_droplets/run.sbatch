#!/bin/bash
# Empty-droplet study on Clipper (bench/empty_droplets). Code comes from this branch, pushed to GitHub, checked out in this
# study's own worktree; data is read from data.singlet.bio inside the job. Nothing here writes to R2 or D1.
#
#   MODE=explore  GSE=...            print a bundle's members and feature names (debug-cpu)
#   MODE=fetch    (array over studies.tsv rows)   fetch the listed samples -> $DATA/<GSE>/<GSM>.npz
#   MODE=analyze  (array over studies.tsv rows)   analyze every fetched sample of that row -> $OUT/<GSE>/<GSM>.json (+ .npz)
#   MODE=all      fetch then analyze
#   MODE=groundtruth GSES="GSE..."   authors' GEO calls and hashtag counts vs ours (after analyze)
#
#   sbatch --partition=debug-cpu --time=00:30:00 --export=ALL,MODE=explore,GSE=GSE163793 run.sbatch
#   sbatch --array=0-16%8 --export=ALL,MODE=all run.sbatch
#SBATCH --job-name=edrop
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BRANCH=${BRANCH:-claude/empty-droplet-check-624070}
REPO=$WORK/repos/singlet-py.git
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
VENV=$WORK/venvs/singlet
BUILD_VENV=$WORK/venvs/v2tools   # has setuptools + pybind11 (used read-only to compile; runs use VENV)
SIF=$WORK/images/python-3.12-bookworm.sif
ZSTD=$WORK/cache/zstd-1.5.6-static
DATA=${DATA:-$WORK/data/emptydrop}
OUT=${OUT:-$WORK/state/emptydrop/results}
MODE=${MODE:-all}
TASK=${SLURM_ARRAY_TASK_ID:-0}
export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip NUMBA_CACHE_DIR=$WORK/cache/numba-edrop MPLCONFIGDIR=$WORK/cache/mpl
SCRATCH=/tmp/${USER:-x}-${SLURM_JOB_ID:-x}-$TASK; mkdir -p "$SCRATCH" "$DATA" "$OUT"; trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }

# ---- code: fetch only this branch into the bare repo, own worktree at the branch tip, in-place _pz build (lock per tip) ----
(
  flock -w 3600 9
  git --git-dir="$REPO" fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
  TIP=$(git --git-dir="$REPO" rev-parse "refs/heads/$BRANCH")
  if [ ! -e "$SRC/.git" ]; then git --git-dir="$REPO" worktree add -q --detach "$SRC" "$TIP"; fi
  if [ "$(git -C "$SRC" rev-parse HEAD)" != "$TIP" ]; then git -C "$SRC" checkout -q --detach "$TIP"; fi
  if ! ls "$SRC"/python/singlet/_pz.cpython-*.so >/dev/null 2>&1 || [ "$(cat "$SRC/.edrop-build" 2>/dev/null)" != "$(git -C "$SRC" rev-parse HEAD:src/bindings/python/_pz.cpp)" ]; then
    rt bash -c "cd '$SRC' && CPATH='$ZSTD/include' LIBRARY_PATH='$ZSTD/lib' '$BUILD_VENV/bin/python' setup.py -q build_ext --inplace -j 8"
    git -C "$SRC" rev-parse HEAD:src/bindings/python/_pz.cpp > "$SRC/.edrop-build"
  fi
) 9>"$WORK/state/edrop-build.lock"
echo "mode=$MODE task=$TASK commit=$(git -C "$SRC" rev-parse --short HEAD) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
PY=("$VENV/bin/python" -P)
B=$SRC/bench/empty_droplets

row() { awk -F'\t' -v t="$TASK" '!/^#/ && NF {if (n==t) {print; exit} n++}' "$B/studies.tsv"; }

case "$MODE" in
  explore)
    rt "${PY[@]}" "$B/fetch.py" "${GSE:?}" --explore --out "$DATA"
    ;;
  fetch|analyze|all)
    R=$(row); [ -n "$R" ] || { echo "no studies.tsv row $TASK"; exit 0; }
    GSE=$(echo "$R" | cut -f1); SEL=$(echo "$R" | cut -f2); echo "row: $R"
    if [ "$MODE" != analyze ]; then
      if [ "$SEL" = all ]; then rt "${PY[@]}" "$B/fetch.py" "$GSE" --out "$DATA"
      elif [[ "$SEL" =~ ^first: ]]; then rt "${PY[@]}" "$B/fetch.py" "$GSE" --first "${SEL#first:}" --out "$DATA"
      else rt "${PY[@]}" "$B/fetch.py" "$GSE" --gsm ${SEL//,/ } --out "$DATA"; fi
    fi
    if [ "$MODE" != fetch ]; then
      rt "${PY[@]}" "$B/analyze.py" "$DATA/$GSE" --out "$OUT/$GSE" --threads "${SLURM_CPUS_PER_TASK:-8}" ${ANALYZE_ARGS:-}
    fi
    ;;
  groundtruth)
    rt "${PY[@]}" "$B/groundtruth.py" ${GSES:?} --data "$DATA" --results "$OUT"
    ;;
  *) echo "MODE must be explore, fetch, analyze, all or groundtruth"; exit 2 ;;
esac
