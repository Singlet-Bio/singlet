#!/bin/bash
# Singlet v2 migration campaign on Clipper: legacy .singlet bundles in R2 -> v2 objects in R2 (scripts/v2_campaign.py).
# One array task = one shard of the studies (accession order). The code is pinned to one pushed commit of feat/singlet-v2;
# the extensions (_pz, _pz2) are built once per commit inside a job (see "Build" below); nothing runs on the login node.
#
#   MODE=build        build + self-test the pinned commit (no R2 access). Run once, on debug-cpu, before anything else.
#   MODE=plan         list the legacy studies, write shard assignment / big-study list, project the wall time (R2 reads)
#                     -> $WORK/state/v2/plan-<c10>/plan.tsv; pass it to dry/full as LISTING=... so every shard slices the same frozen list
#   MODE=dry          shard I of N: download + migrate + verify, NO R2 writes (add LIST_ONLY=1 for HEAD + restricted check only)
#   MODE=pilot        PILOT_N studies spread over the size range (+ GSES="GSE... GSE..."), with writes and SHA-256 read-back
#   MODE=full         shard I of N with writes; safe to resubmit (RESUME=1 skips studies already migrated; RESUME=0 redoes them).
#                     GSES="GSE... GSE..." (with NSHARD=1, array 0) runs just those studies, e.g. after a transcoder fix
#   MODE=summarize    aggregate the results files of RUNS (counts, bytes, throughput, projection)
#   MODE=catalog-sql  SQL batch files for wrangler d1 execute from the rows of RUNS  (pull them with clipper.py get/pull)
#   MODE=catalog-post POST the same rows to the ingest endpoint (needs ~/.config/singlet-ingest.env: INGEST_TOKEN)
#   MODE=refs-index   rebuild v2/ref/index.json (APPLY=1 to write)
#   MODE=retire-dry | retire | restore | audit    scripts/v2_retire.py (retire needs CONFIRM_RETIRE=RETIRE)
#
# Submit (COMMIT = git rev-parse origin/feat/singlet-v2, pushed; see notes/m_w6.md for the full procedure):
#   sbatch --partition=debug-cpu --time=00:30:00 --array=0 --export=ALL,MODE=build,COMMIT=<sha>                 scripts/v2_migrate.sbatch
#   sbatch --array=0       --export=ALL,MODE=pilot,COMMIT=<sha>                                                  scripts/v2_migrate.sbatch
#   sbatch --array=0-31%8  --export=ALL,MODE=dry,NSHARD=32,COMMIT=<sha>                                         scripts/v2_migrate.sbatch
#   sbatch --array=0-31%8  --export=ALL,MODE=full,NSHARD=32,COMMIT=<sha>                                        scripts/v2_migrate.sbatch
#
# Needs: ~/.config/singlet-r2.env (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY), $WORK/images/python-3.12-bookworm.sif
# (the clipper runtime image: compiler, Python headers), the bare clone $WORK/repos/singlet-py.git of singlet-bio.
#SBATCH --job-name=v2mig
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail

WORK=${WORK:-/mnt/projects/debruinz_project/debruinz/singlet}   # overridable only for rehearsals off the cluster
MODE=${MODE:-dry}
BRANCH=${BRANCH:-feat/singlet-v2}
COMMIT=${COMMIT:?set COMMIT to the exact singlet-bio commit to run (git rev-parse origin/feat/singlet-v2, pushed to GitHub)}
C10=${COMMIT:0:10}
REPO=${REPO:-$WORK/repos/singlet-py.git}            # the clipper name of the singlet-bio repo (as in restrict_published_bundles.sbatch)
SRC=$WORK/src/singlet-py/v2mig__$C10                # one worktree + build per commit
VENV=${VENV:-$WORK/venvs/v2tools}
SIF=${SIF:-$WORK/images/python-3.12-bookworm.sif}
ZSTD_VER=1.5.6
ZSTD=$WORK/cache/zstd-$ZSTD_VER-static
TASK=${SLURM_ARRAY_TASK_ID:-0}
RUN=${RUN:-$MODE-$C10}
OUT=$WORK/state/v2/$RUN
NSHARD=${NSHARD:-}
mkdir -p "$WORK/state/v2" "$WORK/cache" "$WORK/logs" "$OUT"

# Keep $HOME clean (100 GB quota): every cache under $WORK, scratch on the node-local disk.
export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
SCRATCH=/tmp/${USER:-$(id -un)}-${SLURM_JOB_ID:-x}-$TASK
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
export TMPDIR=$SCRATCH

# ---- runtime: everything that compiles or runs Python goes through the clipper image ------------------------------
if [ "${RUNTIME:-image}" = host ]; then
  rt() { "$@"; }
else
  command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
  [ -f "$SIF" ] || { echo "runtime image missing: $SIF (run hpc/clipper/setup_runtime.sh)"; exit 2; }
  rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }
fi

time_left() {  # seconds left in this job: squeue's %L is [D-]HH:MM:SS
  local t
  t=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | head -1 || true)
  if [ -z "$t" ] || [ "$t" = UNLIMITED ]; then echo "${TLIMIT_SEC:-14340}"; return; fi
  echo "$t" | awk -F'[-:]' '{ n=NF; s=$n+60*$(n-1); if (n>=3) s+=3600*$(n-2); if (n>=4) s+=86400*$(n-3); print s }'
}

# ---- Build: worktree at the pinned commit, static zstd, venv, extensions (once per commit; array tasks wait on the lock) --
ensure_build() {
  (
    flock -w 7200 9 || { echo "could not get the build lock"; exit 75; }
    [ -e "$SRC/.v2build.ok" ] && exit 0
    echo "[build] commit $COMMIT on $(hostname)"
    if [ ! -e "$SRC/python" ]; then
      git --git-dir="$REPO" fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
      git --git-dir="$REPO" cat-file -e "$COMMIT^{commit}" 2>/dev/null || { echo "commit $COMMIT is not on origin/$BRANCH: push it first"; exit 3; }
      mkdir -p "$(dirname "$SRC")"
      git --git-dir="$REPO" worktree add -q --detach "$SRC" "$COMMIT"
    fi
    [ "$(git -C "$SRC" rev-parse HEAD)" = "$(git --git-dir="$REPO" rev-parse "$COMMIT^{commit}")" ] || { echo "worktree $SRC is not at $COMMIT"; exit 3; }

    # zstd 1.5.6: the nodes have no zstd headers. Build the official source tarball once into a static libzstd.a
    # (-fPIC) shared by every commit; the extensions link it statically, so nothing depends on a library path at run time.
    if [ ! -f "$ZSTD/lib/libzstd.a" ] || [ ! -f "$ZSTD/include/zstd.h" ]; then
      T=$(mktemp -d "$WORK/cache/zstd-build.XXXXXX")
      URL=https://github.com/facebook/zstd/releases/download/v$ZSTD_VER/zstd-$ZSTD_VER.tar.gz
      curl -fsSL --retry 5 --retry-delay 5 -o "$T/zstd.tar.gz" "$URL"
      GOT=$(sha256sum "$T/zstd.tar.gz" | cut -d' ' -f1)
      if curl -fsSL --retry 3 -o "$T/zstd.sha256" "$URL.sha256"; then
        [ "$(cut -d' ' -f1 "$T/zstd.sha256")" = "$GOT" ] || { echo "zstd tarball checksum mismatch ($GOT)"; exit 4; }
      else
        echo "[build] WARNING: no .sha256 published next to the tarball; sha256 is $GOT (pin it here)"
      fi
      echo "[build] zstd-$ZSTD_VER.tar.gz sha256=$GOT"
      tar -xzf "$T/zstd.tar.gz" -C "$T"
      rt bash -c "cd '$T/zstd-$ZSTD_VER/lib' && make -j8 libzstd.a MOREFLAGS=-fPIC ZSTD_LEGACY_SUPPORT=0 >/dev/null"
      mkdir -p "$ZSTD/lib" "$ZSTD/include"
      cp "$T/zstd-$ZSTD_VER/lib/libzstd.a" "$ZSTD/lib/"
      cp "$T/zstd-$ZSTD_VER/lib/zstd.h" "$T/zstd-$ZSTD_VER/lib/zdict.h" "$T/zstd-$ZSTD_VER/lib/zstd_errors.h" "$ZSTD/include/"
      rm -rf "$T"
    fi

    # venv inside the image's Python 3.12, with the package's own runtime dependencies (import singlet is eager), boto3, build tools
    [ -x "$VENV/bin/python" ] || rt python -m venv "$VENV"
    mapfile -t DEPS < <(rt "$VENV/bin/python" -c 'import sys, tomllib; print("\n".join(tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]))' "$SRC/pyproject.toml")
    MARK=$VENV/.deps-$(printf '%s\n' "${DEPS[@]}" boto3 pybind11 setuptools wheel pytest | sha1sum | cut -c1-12)
    if [ ! -e "$MARK" ]; then
      rt "$VENV/bin/pip" install -q "${DEPS[@]}" boto3 pybind11 'setuptools>=64' wheel pytest
      touch "$MARK"
    fi

    # extensions in place: python/singlet/_pz*.so (and _pz2*.so once setup.py lists it)
    rt bash -c "cd '$SRC' && CPATH='$ZSTD/include' LIBRARY_PATH='$ZSTD/lib' '$VENV/bin/python' setup.py -q build_ext --inplace -j 8"
    rt "$VENV/bin/python" - "$SRC" <<'PY'
import importlib, sys
sys.path.insert(0, sys.argv[1] + "/python")
import singlet
from singlet import _pz
print("[build] singlet", singlet.__version__, "_pz ok")
try:
    importlib.import_module("singlet._pz2")
    print("[build] _pz2 ok (native codec)")
except ImportError as e:
    print("[build] WARNING: singlet._pz2 not importable:", e)
PY
    if [ "${REQUIRE_PZ2:-1}" = 1 ]; then
      rt "$VENV/bin/python" -c "import sys; sys.path.insert(0, '$SRC/python'); import singlet._pz2" \
        || { echo "singlet._pz2 is required (REQUIRE_PZ2=0 allows the pure-Python codec: much slower)"; exit 5; }
    fi
    touch "$SRC/.v2build.ok"
  ) 9>"$WORK/state/v2mig-build.lock"
}
[ -e "$SRC/.v2build.ok" ] || ensure_build
[ -e "$SRC/.v2build.ok" ] || { echo "build failed"; exit 6; }

PY=("${PYBIN:-$VENV/bin/python}")
CAMP=("$SRC/scripts/v2_campaign.py")
RET=("$SRC/scripts/v2_retire.py")
export PYTHONPATH=$SRC/python

need_creds() {
  [ -n "${LOCAL_ROOT:-}" ] && return 0   # rehearsal against directories, no R2
  set -a; . "$HOME/.config/singlet-r2.env"; set +a; : "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"
}
need_shard() { [ -n "$NSHARD" ] || { echo "set NSHARD (number of shards = the array size)"; exit 2; }; }
echo "mode=$MODE run=$RUN task=$TASK nshard=${NSHARD:-n/a} commit=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo "$C10") host=$(hostname) scratch=$SCRATCH cpus=${SLURM_CPUS_PER_TASK:-?}"
printf '%s\n' "mode=$MODE commit=$COMMIT task=$TASK nshard=$NSHARD host=$(hostname) at=$(date -u +%FT%TZ)" >> "$OUT/run-info.txt"

# The transcoder and verifier must exist in this commit before a mode that uses them runs.
need_tools() {
  rt "${PY[@]}" -m singlet.v2.migrate --help >/dev/null 2>&1 || { echo "singlet.v2.migrate is not runnable in commit $C10 (W2 deliverable)"; exit 7; }
  rt "${PY[@]}" -m singlet.v2.verify --help >/dev/null 2>&1 || { echo "singlet.v2.verify is not runnable in commit $C10 (W2 deliverable)"; exit 7; }
}

run_campaign() {  # run_campaign <mode> [extra args]
  local mode=$1; shift
  local left; left=$(time_left)
  local deadline=$(( $(date +%s) + left - ${DEADLINE_MARGIN_SEC:-900} ))
  rt "${PY[@]}" "${CAMP[@]}" run --mode "$mode" \
    --results "$OUT/results-$TASK.jsonl" --rows-dir "$OUT/rows" --ledger-dir "$OUT/ledger" --study-log-dir "$OUT/study-logs" \
    --scratch "$SCRATCH/work" --scratch-budget-gb "${SCRATCH_BUDGET_GB:-400}" \
    --workers "${WORKERS:-3}" --migrate-threads "${MIGRATE_THREADS:-2}" --io-threads "${IO_THREADS:-8}" \
    --commit "$COMMIT" --task "$TASK" --deadline-epoch "$deadline" \
    ${EXCLUDE_LIST:+--exclude-list "$EXCLUDE_LIST"} ${LISTING:+--listing "$LISTING"} ${KEEP_SCRATCH:+--keep-scratch "$KEEP_SCRATCH"} \
    ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"} ${MIGRATE_CMD:+--migrate-cmd "$MIGRATE_CMD"} ${VERIFY_CMD:+--verify-cmd "$VERIFY_CMD"} "$@"
}

RUNS=${RUNS:-full-$C10}
rows_args() { local d; for r in $RUNS; do d=$WORK/state/v2/$r/rows; [ -d "$d" ] && printf '%s\n' "$d"; done; }

case "$MODE" in
  build)
    rt "${PY[@]}" -m pytest -q -p no:cacheprovider "$SRC/tests/python/test_v2_campaign.py" "$SRC/tests/python/test_v2_campaign_retire.py" \
      "$SRC/tests/python/test_v2_campaign_publish.py"
    echo "build ok: $SRC"
    ;;
  plan)
    need_creds
    rt "${PY[@]}" "${CAMP[@]}" plan --shards "${NSHARD:-32}" --out "$OUT/plan.tsv" --big-out "$OUT/big.txt" \
      --rate-dl "${RATE_DL:-150}" --rate-cpu "${RATE_CPU:-5}" --cores "${SLURM_CPUS_PER_TASK:-8}" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  dry)
    need_creds; need_shard; need_tools
    run_campaign dry --shard "$TASK/$NSHARD" ${LIST_ONLY:+--list-only} ${GSES:+--gse $GSES}
    ;;
  pilot)
    need_creds; need_tools
    # dedicated run dir per pilot so its rows never mix with the full run's
    run_campaign pilot --pilot-n "${PILOT_N:-6}" --pilot-max-gb "${PILOT_MAX_GB:-8}" ${GSES:+--gse $GSES}
    ;;
  full)
    need_creds; need_shard; need_tools
    extra=(); [ "${RESUME:-1}" = 1 ] && extra+=(--resume)
    run_campaign full --shard "$TASK/$NSHARD" ${extra[@]+"${extra[@]}"} ${GSES:+--gse $GSES}
    ;;
  summarize)
    args=(); for r in $RUNS; do [ -d "$WORK/state/v2/$r" ] && args+=("$WORK/state/v2/$r"); done
    rt "${PY[@]}" "${CAMP[@]}" summarize "${args[@]}" --workers "${WORKERS:-3}"
    ;;
  catalog-sql)
    mapfile -t RD < <(rows_args)
    if [ -n "${FROM_R2:-}" ]; then need_creds; rt "${PY[@]}" "${CAMP[@]}" catalog-sql --from-r2 --out "$OUT/sql" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    else rt "${PY[@]}" "${CAMP[@]}" catalog-sql --rows-dir "${RD[@]}" --out "$OUT/sql"; fi
    ;;
  catalog-post)
    mapfile -t RD < <(rows_args)
    [ -f "$HOME/.config/singlet-ingest.env" ] && { set -a; . "$HOME/.config/singlet-ingest.env"; set +a; }
    rt "${PY[@]}" "${CAMP[@]}" catalog-post --rows-dir "${RD[@]}" ${DRY_RUN:+--dry-run}
    ;;
  refs-index)
    need_creds
    rt "${PY[@]}" "${CAMP[@]}" refs-index ${APPLY:+--apply} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  retire-dry)
    need_creds; need_shard
    rt "${PY[@]}" "${RET[@]}" retire --all --shard "$TASK/$NSHARD" --ledger "$OUT/retire-$TASK.jsonl" ${ALLOW_PARTIAL:+--allow-partial} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  retire)
    need_creds; need_shard
    [ "${CONFIRM_RETIRE:-}" = RETIRE ] || { echo "retiring deletes the public bundles after archiving them: set CONFIRM_RETIRE=RETIRE"; exit 2; }
    rt "${PY[@]}" "${RET[@]}" retire --all --shard "$TASK/$NSHARD" --apply --confirm RETIRE --ledger "$OUT/retire-$TASK.jsonl" ${ALLOW_PARTIAL:+--allow-partial} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  restore)
    need_creds
    [ -n "${GSES:-}" ] || { echo "set GSES=\"GSE... GSE...\""; exit 2; }
    rt "${PY[@]}" "${RET[@]}" restore --gse $GSES ${APPLY:+--apply} --ledger "$OUT/restore-$TASK.jsonl" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  audit)
    need_creds
    rt "${PY[@]}" "${RET[@]}" audit ${EXPECT_NONE:+--expect-none} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  *)
    echo "MODE must be build, plan, dry, pilot, full, summarize, catalog-sql, catalog-post, refs-index, retire-dry, retire, restore or audit"
    exit 2
    ;;
esac
