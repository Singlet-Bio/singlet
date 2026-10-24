#!/bin/bash
# Junction table version 1 on Clipper (docs/format-v2.md section 12; scripts/v2_junction_table.py). Read-only against the public
# data host and the reference FTP sites; writes only under $WORK/state/jxtab/$RUN. Uploading the packs to R2 is a separate,
# reviewed step and is not done here.
#
#   MODE=build    worktree at the pinned commit (jxtab__<c10>), venv, extensions, self-test. Once per commit, on debug-cpu.
#   MODE=list     studies with a legacy bundle (singlet.bio/api/gse) -> studies.tsv
#   MODE=census   array task I of NSHARD: per-junction counts of that shard of studies -> shard_I.parquet + shard_I.json
#   MODE=reduce   shards -> census_<build>.parquet + census_summary.json
#   MODE=table    BUILD=<reference build>: GTF + genome + census -> packs/<build>/ + report_<build>.json
#
#   sbatch --partition=debug-cpu --time=00:30:00 --export=ALL,MODE=build,COMMIT=<sha> scripts/v2_junction_table.sbatch
#   sbatch --partition=debug-cpu --time=00:10:00 --export=ALL,MODE=list,COMMIT=<sha>  scripts/v2_junction_table.sbatch
#   sbatch --array=0-99%25 --export=ALL,MODE=census,NSHARD=100,COMMIT=<sha>          scripts/v2_junction_table.sbatch
#   sbatch --export=ALL,MODE=reduce,COMMIT=<sha> --mem=200G                            scripts/v2_junction_table.sbatch
#   sbatch --export=ALL,MODE=table,BUILD=GRCh38-2024-A,COMMIT=<sha> --mem=120G         scripts/v2_junction_table.sbatch
#SBATCH --job-name=jxtab
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:59:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail

WORK=${WORK:-/mnt/projects/debruinz_project/debruinz/singlet}
MODE=${MODE:?set MODE}
BRANCH=${BRANCH:-feat/v2-junction-table}
COMMIT=${COMMIT:?set COMMIT to the pushed commit of $BRANCH to run}
C10=${COMMIT:0:10}
REPO=$WORK/repos/singlet-py.git
SRC=$WORK/src/singlet-py/jxtab__$C10
VENV=$WORK/venvs/jxtab
SIF=$WORK/images/python-3.12-bookworm.sif
ZSTD=$WORK/cache/zstd-1.5.6-static          # built by the v2 campaign's build step; reused read-only
RUN=${RUN:-v1-$C10}
OUT=$WORK/state/jxtab/$RUN
REFS=$WORK/data/references/junctions
TASK=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$OUT" "$REFS" "$WORK/logs"
export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
SCRATCH=/tmp/${USER:-$(id -un)}-${SLURM_JOB_ID:-x}-$TASK
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH

command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }

if [ "$MODE" = build ]; then
  (
    flock -w 3600 9 || { echo "no build lock"; exit 75; }
    [ -e "$SRC/.jxbuild.ok" ] && exit 0
    [ -f "$ZSTD/lib/libzstd.a" ] || { echo "static zstd missing at $ZSTD (the v2 campaign build makes it)"; exit 4; }
    if [ ! -e "$SRC/python" ]; then
      git --git-dir="$REPO" fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
      git --git-dir="$REPO" cat-file -e "$COMMIT^{commit}" || { echo "commit $COMMIT is not on origin/$BRANCH"; exit 3; }
      mkdir -p "$(dirname "$SRC")"
      git --git-dir="$REPO" worktree add -q --detach "$SRC" "$COMMIT"
    fi
    [ -x "$VENV/bin/python" ] || rt python -m venv "$VENV"
    mapfile -t DEPS < <(rt "$VENV/bin/python" -c 'import sys, tomllib; print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]))' "$SRC/pyproject.toml")
    rt "$VENV/bin/pip" install -q "${DEPS[@]}" pandas pyarrow pybind11 'setuptools>=64' wheel pytest
    rt bash -c "cd '$SRC' && CPATH='$ZSTD/include' LIBRARY_PATH='$ZSTD/lib' CFLAGS=-pthread LDFLAGS=-pthread '$VENV/bin/python' setup.py -q build_ext --inplace -j 8"
    rt bash -c "cd '$SRC' && PYTHONPATH='$SRC/python' '$VENV/bin/python' -m pytest tests/python/test_v2_junctions.py -q -p no:cacheprovider"
    touch "$SRC/.jxbuild.ok"
  ) 9>"$WORK/state/jxtab-build.lock"
  exit 0
fi
[ -e "$SRC/.jxbuild.ok" ] || { echo "run MODE=build for $C10 first"; exit 6; }
PY() { rt env PYTHONPATH="$SRC/python" "$VENV/bin/python" "$SRC/scripts/v2_junction_table.py" "$@"; }
echo "mode=$MODE run=$RUN task=$TASK commit=$C10 host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"

case "$MODE" in
  list)   PY list --out "$OUT" ;;
  census) PY census --out "$OUT" --shard "$TASK" --shards "${NSHARD:?set NSHARD}" --threads "${THREADS:-${SLURM_CPUS_PER_TASK:-8}}" \
            ${STUDIES:+--studies "$STUDIES"} ${SHARDNAME:+--name "$SHARDNAME"} ;;
  reduce) PY reduce --out "$OUT" ;;
  table)  PY build --out "$OUT" --build "${BUILD:?set BUILD}" --refs "$REFS" --census-tag "$RUN" ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
