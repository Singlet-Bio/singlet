#!/bin/bash
#SBATCH --job-name=gf-build
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fast lane build (bench/gpu_fast/README.md). Builds the singlet target only, at a pinned commit, with no ctest.
# Based on bench/gpu_device/jobs/build.sbatch (perf/gpu-device). Do not submit it on the GPU partition.
#
#   sbatch --export=ALL,SHA=<40 hex digits>,NAME=<dir name> bench/gpu_fast/jobs/fast_build.sbatch
#
# Parameters.
#   SHA       required. The full commit. A branch name is refused. The commit is fetched by name under flock when the bare clone lacks it.
#   NAME      required. Output directory $W/jobs/gpu-fast/bin/$NAME with singlet-gpu and commit-gpu. Nothing existing is overwritten.
#   MODE      auto (default), tree, ccache or none.
#               ccache  compiler cache in $W/cache/ccache. Used when a ccache program is found in the toolchain. Fresh build tree.
#               tree    one persistent source tree and build tree in $W/cache/gf-build. The new commit is copied over it file by file and only
#                       files with changed content get a new time stamp, so the build system recompiles those and what depends on them.
#                       Builds are serialized by flock on the tree.
#               none    fresh tree, full build. As build.sbatch.
#             auto picks ccache when found, else tree. The pbench environment has no ccache (checked 2026-11-05), so auto means tree there.
#   PATCH_FILE, PATCH_ID  optional. A throwaway one-line change appended to that source file (relative path) before the build. It exists only in the
#             job's scratch copy. A patched build writes no commit-gpu and no binary. Use it to time a rebuild.
#   MEASURE   1 runs the timing experiment instead of a normal build: clean build, rebuild with nothing changed, rebuild after the patch of
#             PATCH_FILE. It uses a private tree that is deleted afterwards and writes only build-measure.json. Needs PATCH_FILE.
#   EXTRA_CXX C++ flags, as build.sbatch.
# The first log line is BUILD sha=... mode=.... The last lines give seconds per step. build.json in the output directory has the same numbers.
set -euo pipefail
: "${SHA:?SHA: the full 40 digit commit}" "${NAME:?NAME: output directory name}"
[[ $SHA =~ ^[0-9a-f]{40}$ ]] || { echo "SHA must be 40 hex digits, not a branch or a short name"; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] || { echo "NAME: letters digits . _ -"; exit 2; }
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; MODE=${MODE:-auto}; MEASURE=${MEASURE:-0}
case $MODE in auto|tree|ccache|none) ;; *) echo "MODE: auto tree ccache none"; exit 2 ;; esac
if [ "$MEASURE" = 1 ]; then
  : "${PATCH_FILE:?MEASURE needs PATCH_FILE}"
  OUTD=$W/jobs/gpu-fast/build-measure/$NAME
else
  OUTD=$W/jobs/gpu-fast/bin/$NAME
  [ -e $OUTD/singlet-gpu ] && { echo "refusing to overwrite $OUTD"; exit 2; }
fi
mkdir -p $OUTD
[ -z "${PATCH_FILE:-}" ] || [[ $PATCH_FILE =~ ^[A-Za-z0-9_./-]+$ && $PATCH_FILE != *..* ]] || { echo "bad PATCH_FILE"; exit 2; }

# --- the commit, fetched by name under flock ------------------------------------------------------------------------------------------------
URL="$(git -C $BARE config remote.origin.url)"
if ! git -C $BARE cat-file -e "$SHA^{commit}" 2>/dev/null; then
  ( flock 9
    git -C $BARE cat-file -e "$SHA^{commit}" 2>/dev/null || git -C $BARE fetch -q "$URL" '+refs/heads/*:refs/gf/heads/*' || true
    git -C $BARE cat-file -e "$SHA^{commit}" 2>/dev/null || git -C $BARE fetch -q "$URL" "$SHA" || true ) 9>$BARE/.gn-fetch.lock
fi
[ "$(git -C $BARE rev-parse "$SHA^{commit}")" = "$SHA" ] || { echo "commit $SHA is not in $BARE and could not be fetched"; exit 3; }
echo "BUILD sha=$SHA name=$NAME mode=$MODE patch=${PATCH_FILE:-none} measure=$MEASURE job=$SLURM_JOB_ID node=$(hostname -s) $(date -Is)"

SCR=/tmp/$USER-gfb-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf "${SCR:?}"' EXIT
git -C $BARE archive "$SHA" | tar -x -C $SCR/src
S=$SCR/src
# the same edits as build.sbatch: a portable -march and a declaration that old bam_cat.c lacks
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c

# --- compiler cache: is there one? ---------------------------------------------------------------------------------------------------------
CCACHE_BIN=""
for c in $ENV/bin/ccache /usr/bin/ccache /usr/local/bin/ccache $W/cache/ccache-bin/ccache; do [ -x $c ] && { CCACHE_BIN=$c; break; }; done
if [ -z "$CCACHE_BIN" ]; then   # the CUDA image may carry one
  apptainer exec -B $W $IMG sh -c 'command -v ccache' > $SCR/cc.txt 2>/dev/null && CCACHE_BIN=$(cat $SCR/cc.txt) || true
fi
RES=$MODE
if [ $MODE = auto ]; then [ -n "$CCACHE_BIN" ] && RES=ccache || RES=tree; fi
[ $RES != ccache ] || [ -n "$CCACHE_BIN" ] || { echo "MODE=ccache but no ccache program was found. Install one (conda create -p $W/envs/ccache -c conda-forge ccache) or use MODE=tree."; exit 2; }
echo "compiler cache: ${CCACHE_BIN:-none found}. Resolved mode: $RES"

# --- source and build directories ------------------------------------------------------------------------------------------------------------
sync_tree() {   # copy $1 over $2: write only files whose content changed, delete files that are gone
  $ENV/bin/python3 - "$1" "$2" <<'PYEOF'
import os, shutil, sys
src, dst = sys.argv[1:3]
n_copy = n_del = 0
seen = set()
for root, dirs, files in os.walk(src):
    rel = os.path.relpath(root, src)
    os.makedirs(os.path.join(dst, rel), exist_ok=True)
    for f in files:
        p, q = os.path.join(root, f), os.path.normpath(os.path.join(dst, rel, f))
        seen.add(q)
        if os.path.islink(p):
            if os.path.lexists(q): os.remove(q)
            os.symlink(os.readlink(p), q); n_copy += 1; continue
        if os.path.exists(q) and os.path.getsize(q) == os.path.getsize(p) and open(p, "rb").read() == open(q, "rb").read():
            continue
        shutil.copyfile(p, q); shutil.copymode(p, q); n_copy += 1
for root, dirs, files in os.walk(dst):
    for f in files:
        q = os.path.normpath(os.path.join(root, f))
        if q not in seen: os.remove(q); n_del += 1
print(f"sync: {n_copy} files written, {n_del} removed")
PYEOF
}
case $RES in
  tree)
    CACHE=${CACHE:-$W/cache/gf-build}; [ "$MEASURE" = 1 ] && CACHE=$W/cache/gf-measure-$SLURM_JOB_ID
    mkdir -p $CACHE; SRC=$CACHE/src; BUILD=$CACHE/build; mkdir -p $SRC $BUILD
    exec 8>$CACHE/.lock; echo "waiting for the tree lock"; flock 8 ;;
  ccache)
    export CCACHE_DIR=$W/cache/ccache CCACHE_MAXSIZE=20G CCACHE_SLOPPINESS=time_macros CCACHE_BASEDIR=$SCR; mkdir -p $CCACHE_DIR
    SRC=$S; BUILD=$SCR/b ;;
  none) SRC=$S; BUILD=$SCR/b ;;
esac
LAUNCH=""; [ $RES = ccache ] && LAUNCH="-DCMAKE_C_COMPILER_LAUNCHER=$CCACHE_BIN -DCMAKE_CXX_COMPILER_LAUNCHER=$CCACHE_BIN -DCMAKE_CUDA_COMPILER_LAUNCHER=$CCACHE_BIN"

cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
export CCACHE_DIR="${CCACHE_DIR:-}" CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-}" CCACHE_SLOPPINESS="${CCACHE_SLOPPINESS:-}" CCACHE_BASEDIR="${CCACHE_BASEDIR:-}"
cmake -S $SRC -B $BUILD -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$SHA -DSINGLET_ENGINE_GPU=ON -DCMAKE_CXX_FLAGS="${EXTRA_CXX:-}" \
  -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="90" $LAUNCH > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $BUILD --target singlet -j ${SLURM_CPUS_PER_TASK:-24} > $SCR/b.log 2>&1 || { grep -m40 -B3 -A10 "error" $SCR/b.log || tail -60 $SCR/b.log; exit 1; }
${CCACHE_BIN:-true} -s 2>/dev/null | head -12 || true
EOS

TIMES=""
one_build() {   # label: copy the tree, build, record seconds
  local label=$1 t0=$SECONDS
  if [ $RES = tree ]; then sync_tree $S $SRC | tee -a $SCR/sync.log; fi
  local t1=$SECONDS
  apptainer exec -B $W -B $SCR -B ${CACHE:-$SCR} $IMG bash $SCR/build.sh
  local t2=$SECONDS
  cp $SCR/b.log $OUTD/build-$label.log 2>/dev/null || true
  local nobj; nobj=$(grep -cE "Building (C|CXX|CUDA) object" $SCR/b.log || true)
  TIMES="$TIMES\"$label\":{\"sync_s\":$((t1 - t0)),\"build_s\":$((t2 - t1)),\"objects_compiled\":${nobj:-0}},"
  echo "[build] $label: sync $((t1 - t0)) s, configure and build $((t2 - t1)) s, $nobj objects compiled (mode $RES)"
}
patch_tree() {   # the throwaway change: it changes the preprocessed text, so a compiler cache cannot hit it
  printf '\nnamespace { [[maybe_unused]] const int gf_probe_%s = %s; }\n' "${PATCH_ID:-1}" "${PATCH_ID:-1}" >> $S/$PATCH_FILE
}

if [ "$MEASURE" = 1 ]; then
  one_build clean
  one_build nochange
  patch_tree; one_build rebuild_after_one_file_patch
  printf '{"sha":"%s","mode":"%s","patch_file":"%s","job":"%s",%s"node":"%s"}\n' $SHA $RES "$PATCH_FILE" $SLURM_JOB_ID "$TIMES" "$(hostname -s)" > $OUTD/build-measure.json
  exec 8>&-
  [ $RES != tree ] || rm -rf $CACHE || true
  cat $OUTD/build-measure.json
  exit 0
fi
PATCHED=0
if [ -n "${PATCH_FILE:-}" ]; then patch_tree; PATCHED=1; fi
one_build build
if [ $PATCHED = 1 ]; then
  echo "[build] patched build: no commit-gpu and no binary are written"
  printf '{"sha":"%s","mode":"%s","patched":true,"job":"%s",%s"node":"%s"}\n' $SHA $RES $SLURM_JOB_ID "$TIMES" "$(hostname -s)" > $OUTD/build.json
  exit 0
fi
BIN=$BUILD/src/pipeline/singlet
cp $BIN $OUTD/singlet-gpu
echo "$SHA" > $OUTD/commit-gpu
grep -aqF "$SHA" $OUTD/singlet-gpu && EMB=yes || EMB=no
ln -sfn $W/pipeline-benchmark/whitelists $OUTD/../whitelists 2>/dev/null || true
printf '{"sha":"%s","mode":"%s","patched":false,"embedded":"%s","job":"%s",%s"node":"%s"}\n' $SHA $RES $EMB $SLURM_JOB_ID "$TIMES" "$(hostname -s)" > $OUTD/build.json
sha256sum $OUTD/singlet-gpu; ldd $OUTD/singlet-gpu | grep -iE "cuda|not found" || true
echo "[build] $NAME ok, commit embedded: $EMB, $TIMES"
[ $EMB = yes ] || { echo "the commit string is not inside the binary"; exit 4; }
