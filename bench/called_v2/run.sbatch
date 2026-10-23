#!/bin/bash
# called_v2 validation on Clipper (bench/called_v2). Reads the samples the empty-droplet study fetched ($WORK/data/emptydrop,
# from data.singlet.bio) and that study's profile-test results; ground truth comes from GEO inside the job. Writes nothing to
# R2 or D1.
#
#   MODE=validate (array over studies.tsv rows of bench/empty_droplets)  -> $OUT/<GSE>/<GSM>.v2.json + .pv.npz
#   MODE=inspect                                                       outlier breakdown (log)
#   MODE=aggregate                                                     -> $OUT/../report
#
#   sbatch --array=0-17%9 --export=ALL,MODE=validate run.sbatch
#SBATCH --job-name=cv2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BRANCH=${BRANCH:-claude/festive-vaughan-4a4a8f}
REPO=$WORK/repos/singlet-py.git
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
VENV=$WORK/venvs/singlet
SIF=$WORK/images/python-3.12-bookworm.sif
DATA=${DATA:-$WORK/data/emptydrop}
EDROP=${EDROP:-$WORK/state/emptydrop/results}
OUT=${OUT:-$WORK/state/called_v2/results}
CACHE=${CACHE:-$WORK/data/called_v2/truth}
MODE=${MODE:-validate}
TASK=${SLURM_ARRAY_TASK_ID:-0}
export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip NUMBA_CACHE_DIR=$WORK/cache/numba-cv2 MPLCONFIGDIR=$WORK/cache/mpl
SCRATCH=/tmp/${USER:-x}-${SLURM_JOB_ID:-x}-$TASK; mkdir -p "$SCRATCH" "$OUT" "$CACHE"; trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }

# code: fetch only this branch into the bare repo and check out its tip in this study's own worktree (no build needed)
(
  flock -w 3600 9
  git --git-dir="$REPO" fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
  TIP=$(git --git-dir="$REPO" rev-parse "refs/heads/$BRANCH")
  if [ ! -e "$SRC/.git" ]; then git --git-dir="$REPO" worktree add -q --detach "$SRC" "$TIP"; fi
  if [ "$(git -C "$SRC" rev-parse HEAD)" != "$TIP" ]; then git -C "$SRC" checkout -q --detach "$TIP"; fi
) 9>"$WORK/state/cv2-build.lock"
echo "mode=$MODE task=$TASK commit=$(git -C "$SRC" rev-parse --short HEAD) host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
PY=("$VENV/bin/python" -P)
B=$SRC/bench/called_v2

case "$MODE" in
  validate)
    GSE=$(awk -F'\t' -v t="$TASK" '!/^#/ && NF {if (n==t) {print $1; exit} n++}' "$SRC/bench/empty_droplets/studies.tsv")
    [ -n "$GSE" ] || { echo "no studies.tsv row $TASK"; exit 0; }
    [ -d "$DATA/$GSE" ] || { echo "no fetched data for $GSE"; exit 0; }
    rt "${PY[@]}" "$B/validate.py" "$DATA/$GSE" --out "$OUT/$GSE" --truth-cache "$CACHE" --edrop-results "$EDROP/$GSE" \
      --threads "${SLURM_CPUS_PER_TASK:-8}" ${VALIDATE_ARGS:-}
    ;;
  inspect)
    rt "${PY[@]}" "$B/inspect_outliers.py" --data "$DATA" --results "$OUT" --edrop "$EDROP" --truth-cache "$CACHE"
    ;;
  aggregate)
    rt "${PY[@]}" "$B/aggregate.py" "$OUT" --out "${REPORT:-$WORK/state/called_v2/report}"
    ;;
  *) echo "MODE must be validate, inspect or aggregate"; exit 2 ;;
esac
