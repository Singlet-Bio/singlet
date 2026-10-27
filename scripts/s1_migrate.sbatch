#!/bin/bash
# Singlet .s1 migration campaign on Clipper: legacy .singlet bundles in R2 -> .s1 objects in R2 (scripts/s1_campaign.py).
# One array task = one shard of the studies (accession order). The code is pinned to one pushed commit of feat/singlet-v2;
# the extensions (_pz, _pz2) are built once per commit inside a job (see "Build" below); nothing runs on the login node.
#
#   MODE=build        build + self-test the pinned commit (no R2 access). Run once, on debug-cpu, before anything else.
#   MODE=plan         list the legacy studies; write the frozen listing, the big-study list and a time projection (R2 reads)
#                     -> $WORK/state/s1/plan-<c10>/{plan.tsv,big.txt}; give plan.tsv to dry/full as LISTING=...
#   MODE=dry          shard I of N: download + migrate + verify, NO R2 writes (LIST_ONLY=1: HEAD + restricted check only)
#   MODE=pilot        PILOT_N studies spread over the size range (+ GSES="GSE... GSE..."), with writes and SHA-256 read-back
#   MODE=full         shard I of N with writes; safe to resubmit (RESUME=1 skips studies already migrated; RESUME=0 redoes them).
#                     GSES="GSE... GSE..." (NSHARD=1, array 0) runs just those studies; GSE_LIST=file likewise;
#                     EXCLUDE_LIST=big.txt leaves the big studies to their own job (see notes/m_w6.md)
#   MODE=summarize    aggregate the results files of RUNS (counts, bytes, per-stage throughput, projection)
#   MODE=catalog-sql  SQL files (schema 032) from the rows of RUNS, in days of at most DAY_BUDGET (80000) estimated row writes
#                     (FROM_R2=1: rebuild the rows from the manifests and ledgers in R2). The orchestrator applies them with wrangler.
#   MODE=catalog-post POST the manifests to /api/ingest/s1-study (needs ~/.config/singlet-ingest.env: INGEST_TOKEN; secondary route)
#   MODE=refs-index   publish the reference packs and s1/ref/index.json from python/singlet/s1/packs (APPLY=1 to write). Run before the campaign.
#   Retirement, three separate steps (notes/m_w6.md section 6), so the site never offers a dead legacy URL:
#   MODE=archive-dry | archive   (A) copy data/<GSE>/<GSE>.singlet to singlet-archive, verify, write archived-<task>.jsonl; deletes nothing
#   MODE=retire-sql              (B) SQL flipping s1_migration.status verified -> retired for the archived studies (the orchestrator applies it)
#   MODE=delete-dry | delete     (C) delete from singlet-data only for archived studies that D1 shows retired >= 5 min and verified >= 24 h:
#                                needs AGES=<export of the retired rows>, delete needs CONFIRM_RETIRE=RETIRE
#   MODE=restore | audit
#
# Submit (COMMIT = git rev-parse origin/feat/singlet-v2, pushed; notes/m_w6.md has the full procedure and the numbers):
#   sbatch --partition=debug-cpu --time=00:30:00 --array=0 --export=ALL,MODE=build,COMMIT=<sha>   scripts/s1_migrate.sbatch
#   sbatch --array=0       --export=ALL,MODE=pilot,COMMIT=<sha>                                    scripts/s1_migrate.sbatch
#   sbatch --array=0-63%12 --export=ALL,MODE=full,NSHARD=64,LISTING=<plan.tsv>,EXCLUDE_LIST=<big.txt>,COMMIT=<sha>   scripts/s1_migrate.sbatch
#
# Research Alpha (docs/s1-migration.md, Research Alpha): ALPHA_POLICY=<policy json> (withheld studies skipped; excluded samples and withheld
# layers never written; policy audit per study in $OUT/policy-audit/, private), SAMPLE_FACTS=<json> (provenance), LABELS=1 (per-cell
# labels from the decoded counts into s1/a/; AF=<1kgp af parquet> for ancestry, RESTRICTED_BUCKET for the depth tables),
# ALPHA_POLICY_SHA256=<hex> (refuse a policy file with other bytes). READBACK=sha256 re-reads every uploaded object. BRANCH=<the branch COMMIT is on>.
# Needs: ~/.config/singlet-r2.env (R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY), $WORK/images/python-3.12-bookworm.sif
# (the clipper runtime image: compiler, Python headers), the bare clone $WORK/repos/singlet-py.git of singlet-bio.
# A migrate process is one Python thread (its native encoder adds 4 threads for a few seconds per sample): a job runs
# WORKERS = cpus - 2 studies at once. Memory: about 2 to 4 GB per worker.
#SBATCH --job-name=stmig
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --no-requeue
# (scavenger runs pass --requeue: a full-mode task is safe to requeue, RESUME=1 skips every study already committed)
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
ZSTD_VER=1.5.6                                      # the zstd the legacy writer calibration (ccd0ac25) is known for; 1.5.7 writes other frames
ZSTD=$WORK/cache/zstd-$ZSTD_VER-static
TASK=${SLURM_ARRAY_TASK_ID:-0}
RUN=${RUN:-$MODE-$C10}
OUT=$WORK/state/s1/$RUN
NSHARD=${NSHARD:-}
mkdir -p "$WORK/state/s1" "$WORK/cache" "$WORK/logs" "$OUT"

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

# ---- Build: worktree at the pinned commit, static zstd, venv, extensions, self-test (once per commit; array tasks wait on the lock) --
# What the build needs, from setup.py of the commit: Pybind11Extension singlet._pz (src/bindings/python/_pz.cpp) and singlet._pz2
# (_pz2.cpp, include_dirs include + src/bindings/common), both C++17, libraries=["zstd"], no -march (pz2 picks AVX2/AVX-512 at run
# time from cpuid, with per-function target attributes), std::thread in the encoder (-pthread), Python headers, pybind11, a C++17
# compiler (the image's GCC 12), and zstd.h + libzstd.a (built below). Nothing else.
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
    # data files the transcoder reads at run time must be in the commit (they are easy to leave untracked)
    ls "$SRC"/python/singlet/s1/packs/index.json "$SRC"/python/singlet/s1/packs/*.pack >/dev/null \
      || { echo "commit $C10 has no python/singlet/s1/packs/*.pack + index.json (git add them)"; exit 8; }
    ls "$SRC"/python/singlet/s1/recipes_data/* >/dev/null 2>&1 || { echo "commit $C10 has no python/singlet/s1/recipes_data/ (git add it)"; exit 8; }
    [ -f "$SRC/src/bindings/python/_pz2.cpp" ] && grep -q "singlet._pz2" "$SRC/setup.py" \
      || { echo "commit $C10 does not build singlet._pz2 (src/bindings/python/_pz2.cpp + setup.py)"; exit 8; }

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
    MARK=$VENV/.deps-$(printf '%s\n' "${DEPS[@]}" numba boto3 pybind11 setuptools wheel pytest | sha1sum | cut -c1-12)
    if [ ! -e "$MARK" ]; then
      rt "$VENV/bin/pip" install -q "${DEPS[@]}" 'numba>=0.57' boto3 pybind11 'setuptools>=64' wheel pytest
      touch "$MARK"
    fi

    # extensions in place: python/singlet/_pz*.so and _pz2*.so. CPATH/LIBRARY_PATH find the static zstd; -pthread for std::thread.
    rt bash -c "cd '$SRC' && CPATH='$ZSTD/include' LIBRARY_PATH='$ZSTD/lib' CFLAGS=-pthread LDFLAGS=-pthread '$VENV/bin/python' setup.py -q build_ext --inplace -j 8"
    rt "$VENV/bin/python" - "$SRC" <<'PY'
import sys
sys.path.insert(0, sys.argv[1] + "/python")
import singlet
from singlet import _pz, _pz2
print("[build] singlet", singlet.__version__, "_pz and _pz2 import")
from singlet.s1 import recipes_pzw
ids = sorted(recipes_pzw.available_writers())
print("[build] legacy .1pz writers (calibration ids):", ids, "(ccd0ac25 = zstd 1.5.5/1.5.6)")
if "ccd0ac25" not in ids:
    print("[build] WARNING: no writer with calibration ccd0ac25: derived .1pz members will report container bytes as not reproducible (informational, D15)")
PY

    # self-test of the container, the transcoder and the verifier on this node, before any task may use the build
    rt bash -c "cd '$SRC' && PYTHONPATH='$SRC/python' SINGLET_PZW_AUTOBUILD=0 '$VENV/bin/python' -m pytest tests/python/test_s1_container.py -q -p no:cacheprovider"
    rt bash -c "cd '$SRC' && export PYTHONPATH='$SRC/python' && '$VENV/bin/python' -m singlet.s1.migrate --help >/dev/null && '$VENV/bin/python' -m singlet.s1.verify --help >/dev/null"
    touch "$SRC/.v2build.ok"
  ) 9>"$WORK/state/stmig-build.lock"
}
[ -e "$SRC/.v2build.ok" ] || ensure_build
[ -e "$SRC/.v2build.ok" ] || { echo "build failed"; exit 6; }

PY=("${PYBIN:-$VENV/bin/python}")
CAMP=("$SRC/scripts/s1_campaign.py")
RET=("$SRC/scripts/s1_retire.py")
export PYTHONPATH=$SRC/python

need_creds() {
  [ -n "${LOCAL_ROOT:-}" ] && return 0   # rehearsal against directories, no R2
  set -a; . "$HOME/.config/singlet-r2.env"; set +a; : "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"
}
if [ -n "${ALPHA_POLICY:-}" ]; then
  [ -f "$ALPHA_POLICY" ] || { echo "ALPHA_POLICY $ALPHA_POLICY not found"; exit 2; }
  GOT=$(sha256sum "$ALPHA_POLICY" | cut -d' ' -f1)
  if [ -n "${ALPHA_POLICY_SHA256:-}" ] && [ "$GOT" != "$ALPHA_POLICY_SHA256" ]; then echo "policy sha256 $GOT != $ALPHA_POLICY_SHA256"; exit 2; fi
  echo "release policy: $ALPHA_POLICY sha256=$GOT"
fi
AF=${AF:-$WORK/data/annotate/1kgp_superpop_af.parquet}
[ -f "$AF" ] || AF=
need_shard() { [ -n "$NSHARD" ] || { echo "set NSHARD (number of shards = the array size)"; exit 2; }; }
echo "mode=$MODE run=$RUN task=$TASK nshard=${NSHARD:-n/a} commit=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo "$C10") host=$(hostname) scratch=$SCRATCH cpus=${SLURM_CPUS_PER_TASK:-?}"
printf '%s\n' "mode=$MODE commit=$COMMIT task=$TASK nshard=$NSHARD host=$(hostname) at=$(date -u +%FT%TZ)" >> "$OUT/run-info.txt"

# The transcoder and verifier must run in this commit before a mode that uses them starts.
need_tools() {
  rt "${PY[@]}" -m singlet.s1.migrate --help >/dev/null 2>&1 || { echo "singlet.s1.migrate is not runnable in commit $C10"; exit 7; }
  rt "${PY[@]}" -m singlet.s1.verify --help >/dev/null 2>&1 || { echo "singlet.s1.verify is not runnable in commit $C10"; exit 7; }
}

run_campaign() {  # run_campaign <mode> [extra args]
  local mode=$1; shift
  local left; left=$(time_left)
  local deadline=$(( $(date +%s) + left - ${DEADLINE_MARGIN_SEC:-900} ))
  rt "${PY[@]}" "${CAMP[@]}" run --mode "$mode" \
    --results "$OUT/results-$TASK.jsonl" --rows-dir "$OUT/rows" --ledger-dir "$OUT/ledger" --study-log-dir "$OUT/study-logs" \
    --scratch "$SCRATCH/work" --scratch-budget-gb "${SCRATCH_BUDGET_GB:-400}" \
    ${WORKERS:+--workers "$WORKERS"} --io-threads "${IO_THREADS:-8}" --min-rate-mbs "${MIN_RATE_MBS:-0.4}" \
    --commit "$COMMIT" --task "$TASK" --deadline-epoch "$deadline" \
    ${EXCLUDE_LIST:+--exclude-list "$EXCLUDE_LIST"} ${LISTING:+--listing "$LISTING"} ${GSE_LIST:+--gse-list "$GSE_LIST"} \
    ${KEEP_SCRATCH:+--keep-scratch "$KEEP_SCRATCH"} ${READBACK:+--readback "$READBACK"} ${MIGRATE_TIMEOUT:+--migrate-timeout "$MIGRATE_TIMEOUT"} ${VERIFY_TIMEOUT:+--verify-timeout "$VERIFY_TIMEOUT"} \
    ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"} ${MIGRATE_CMD:+--migrate-cmd "$MIGRATE_CMD"} ${VERIFY_CMD:+--verify-cmd "$VERIFY_CMD"}     ${ALPHA_POLICY:+--alpha-policy "$ALPHA_POLICY" --audit-dir "$OUT/policy-audit"} ${SAMPLE_FACTS:+--sample-facts "$SAMPLE_FACTS"}     ${LABELS:+--labels --labels-timeout "${LABELS_TIMEOUT:-1800}" --labels-max-gb "${LABELS_MAX_GB:-24}"}     ${LABELS:+${AF:+--af "$AF"}} ${LABELS:+--restricted-bucket "${RESTRICTED_BUCKET:-singlet-restricted}"} "$@"
}

rows_args() { local d; for r in ${RUNS:-full-$C10}; do d=$WORK/state/s1/$r/rows; [ -d "$d" ] && printf '%s\n' "$d"; done; }

case "$MODE" in
  build)
    rt "${PY[@]}" -m pytest -q -p no:cacheprovider "$SRC/tests/python/test_s1_campaign.py" "$SRC/tests/python/test_s1_campaign_retire.py" \
      "$SRC/tests/python/test_s1_campaign_publish.py" "$SRC/tests/python/test_s1_campaign_catalog.py"
    echo "build ok: $SRC"
    ;;
  plan)
    need_creds
    rt "${PY[@]}" "${CAMP[@]}" plan --shards "${NSHARD:-64}" --out "$OUT/plan.tsv" --big-out "$OUT/big.txt" --big-gb "${BIG_GB:-5}" \
      --rate-dl "${RATE_DL:-150}" --rate-cpu "${RATE_CPU:-1.0}" --cores "${SLURM_CPUS_PER_TASK:-16}" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  dry)
    need_creds; need_shard; need_tools
    run_campaign dry --shard "$TASK/$NSHARD" ${LIST_ONLY:+--list-only} ${GSES:+--gse $GSES}
    ;;
  pilot)
    need_creds; need_tools
    # dedicated run dir per pilot so its rows never mix with the full run's
    run_campaign pilot --pilot-n "${PILOT_N:-6}" --pilot-max-gb "${PILOT_MAX_GB:-2}" ${GSES:+--gse $GSES}
    ;;
  full)
    need_creds; need_shard; need_tools
    extra=(); [ "${RESUME:-1}" = 1 ] && extra+=(--resume)
    run_campaign full --shard "$TASK/$NSHARD" ${extra[@]+"${extra[@]}"} ${GSES:+--gse $GSES}
    ;;
  summarize)
    args=(); for r in ${RUNS:-full-$C10}; do [ -d "$WORK/state/s1/$r" ] && args+=("$WORK/state/s1/$r"); done
    rt "${PY[@]}" "${CAMP[@]}" summarize "${args[@]}" --workers "${WORKERS:-$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))}"
    ;;
  catalog-sql)
    mapfile -t RD < <(rows_args)
    if [ -n "${FROM_R2:-}" ]; then need_creds; rt "${PY[@]}" "${CAMP[@]}" catalog-sql --from-r2 --out "$OUT/sql" --day-budget "${DAY_BUDGET:-80000}" --schema "${D1_SCHEMA:-033}" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    else rt "${PY[@]}" "${CAMP[@]}" catalog-sql --rows-dir "${RD[@]}" --out "$OUT/sql" --day-budget "${DAY_BUDGET:-80000}" --schema "${D1_SCHEMA:-033}"; fi
    # one compressed file to bring back to the laptop (the whole atlas is about 200 MB of SQL, about 25 MB packed)
    tar --force-local -C "$OUT" -czf "$OUT/sql.tar.gz" sql && ls -l "$OUT/sql.tar.gz"
    ;;
  catalog-post)
    need_creds
    [ -f "$HOME/.config/singlet-ingest.env" ] && { set -a; . "$HOME/.config/singlet-ingest.env"; set +a; }
    if [ -n "${GSES:-}" ]; then printf '%s\n' $GSES > "$SCRATCH/gses.txt"; fi
    rt "${PY[@]}" "${CAMP[@]}" catalog-post ${GSES:+--gse-list "$SCRATCH/gses.txt"} ${DRY_RUN:+--dry-run} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  refs-index)
    need_creds
    rt "${PY[@]}" "${CAMP[@]}" refs-index ${APPLY:+--apply} ${MERGE_PREFIX:+--merge-prefix "$MERGE_PREFIX"} --scratch "$SCRATCH" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  archive-dry)  # (A) read-only: what would be copied
    need_creds; need_shard
    rt "${PY[@]}" "${RET[@]}" archive --all --shard "$TASK/$NSHARD" --ledger "$OUT/archive-$TASK.jsonl" ${ALLOW_PARTIAL:+--allow-partial} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  archive)  # (A) server-side copy to singlet-archive, verified, written to $OUT/archived-<task>.jsonl. Deletes nothing.
    need_creds; need_shard
    rt "${PY[@]}" "${RET[@]}" archive --all --shard "$TASK/$NSHARD" --apply --list "$OUT/archived-$TASK.jsonl" --ledger "$OUT/archive-$TASK.jsonl" ${ALLOW_PARTIAL:+--allow-partial} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  retire-sql)  # (B) SQL flipping s1_migration.status verified -> retired for the archived studies of RUNS (default archive-<c10>)
    args=(); for r in ${RUNS:-archive-$C10}; do for f in "$WORK"/state/s1/$r/archived-*.jsonl "$WORK"/state/s1/$r/restored-*.jsonl; do [ -f "$f" ] && args+=("$f"); done; done
    [ ${#args[@]} -gt 0 ] || { echo "no archived-*.jsonl in RUNS=${RUNS:-archive-$C10}"; exit 2; }
    rt "${PY[@]}" "${CAMP[@]}" retire-sql --list "${args[@]}" --out "$OUT/sql"
    tar --force-local -C "$OUT" -czf "$OUT/sql.tar.gz" sql && ls -l "$OUT/sql.tar.gz"
    ;;
  delete-dry|delete)  # (C) delete the legacy objects of archived studies whose catalog row has been retired long enough (AGES from D1)
    need_creds; need_shard
    [ -n "${AGES:-}" ] && [ -f "$AGES" ] || { echo "set AGES=<csv/json exported from D1: gse_id, verified_at, updated_at of the rows with status = 'retired'> (clipper.py put it first)"; exit 2; }
    if [ -z "${ARCHIVED:-}" ]; then  # the shard lists of the archive run(s), plus any restore lists (a later restore cancels an archive)
      ARCHIVED=$SCRATCH/archived.jsonl; : > "$ARCHIVED"
      for r in ${RUNS:-archive-$C10}; do for f in "$WORK"/state/s1/$r/archived-*.jsonl "$WORK"/state/s1/$r/restored-*.jsonl; do [ -f "$f" ] && cat "$f" >> "$ARCHIVED"; done; done
    fi
    [ -s "$ARCHIVED" ] || { echo "the archived list is empty"; exit 2; }
    flags=()
    if [ "$MODE" = delete ]; then
      [ "${CONFIRM_RETIRE:-}" = RETIRE ] || { echo "this deletes public bundles: set CONFIRM_RETIRE=RETIRE"; exit 2; }
      flags=(--apply --confirm RETIRE)
    fi
    rt "${PY[@]}" "${RET[@]}" delete --list "$ARCHIVED" --ages "$AGES" --shard "$TASK/$NSHARD" \
      --min-verified-age-hours "${MIN_VERIFIED_AGE_HOURS:-24}" --min-retired-age-minutes "${MIN_RETIRED_AGE_MINUTES:-5}" \
      --ledger "$OUT/delete-$TASK.jsonl" ${NOW:+--now "$NOW"} ${flags[@]+"${flags[@]}"} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  restore)
    need_creds
    [ -n "${GSES:-}" ] || { echo "set GSES=\"GSE... GSE...\""; exit 2; }
    rt "${PY[@]}" "${RET[@]}" restore --gse $GSES ${APPLY:+--apply} --list "$OUT/restored-$TASK.jsonl" --ledger "$OUT/restore-$TASK.jsonl" ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  audit)
    need_creds
    rt "${PY[@]}" "${RET[@]}" audit ${EXPECT_NONE:+--expect-none} ${LOCAL_ROOT:+--local-root "$LOCAL_ROOT"}
    ;;
  *)
    echo "MODE must be build, plan, dry, pilot, full, summarize, catalog-sql, catalog-post, refs-index, archive-dry, archive, retire-sql, delete-dry, delete, restore or audit"
    exit 2
    ;;
esac
