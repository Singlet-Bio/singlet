#!/bin/bash
#SBATCH --job-name=rt-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-build: singlet (--engine gpu) from a commit of the bare clone plus a patch, for a timing arm that is not on a pushed branch yet.
# The recipe is jobs/gpu-campaign/gc-build.sbatch (pbench toolchain in the CUDA 12.4 devel image); the differences: the source is
# BASE (a commit or ref already in $W/repos/singlet-py.git, nothing is fetched) with PATCH (a `git diff` file) applied by `git apply`.
#   BASE=<commit> PATCH=<file> NAME=<output dir name> [TESTS=1|only] sbatch rt_build.sbatch
# TESTS=1 also builds the CPU unit tests in the container and runs them on the host (the conda toolchain links test binaries against
# GLIBC_2.38, which the container lacks, and the pz2 tests need the conda zstd headers: jobs/gpu-pr2/ctest-host2.sbatch); TESTS=only
# does only that, into an existing or new $NAME directory.
# Output: $W/jobs/gpu-runtime/bin/$NAME/singlet-gpu, BUILD.txt (base commit, patch sha256, CUDA archs, nvcc, sha256), the patch itself.
set -euo pipefail
: "${BASE:?}" "${PATCH:?}" "${NAME:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
TESTS=${TESTS:-0}
BARE=$W/repos/singlet-py.git; OUTD=$W/jobs/gpu-runtime/bin/$NAME; [ "$TESTS" != only ] && [ -e $OUTD/singlet-gpu ] && { echo "refusing to overwrite $OUTD"; exit 2; }; mkdir -p $OUTD
COMMIT=$(git -C $BARE rev-parse --verify "$BASE^{commit}"); echo "BASE=$COMMIT NAME=$NAME PATCH=$PATCH"
SCR=/tmp/$USER-rtb-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src
S=$SCR/src
( cd $S && git apply --stat --apply -p1 "$PATCH" ) || { echo "[build] the patch does not apply to $COMMIT"; exit 3; }
cp "$PATCH" $OUTD/patch.diff
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
\$CXX --version | head -1; nvcc --version | tail -1
if [ "$TESTS" != only ]; then
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=ON -DCMAKE_CXX_FLAGS="${EXTRA_CXX:-}" \
  -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-70;75;90}" > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target singlet -j ${SLURM_CPUS_PER_TASK:-24} > $SCR/b.log 2>&1 || { grep -m40 -B3 -A10 "error" $SCR/b.log || tail -60 $SCR/b.log; exit 1; }
grep -E "allsite_gpu.*warning" $SCR/b.log | head -20 || true
cp $SCR/b/src/pipeline/singlet $OUTD/singlet-gpu
cuobjdump --list-elf $OUTD/singlet-gpu 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '
' ' ' > $SCR/archs.txt || true
nvcc --version | tail -1 > $SCR/nvcc.txt
fi
if [ "$TESTS" != 0 ]; then   # the CPU unit tests of the same tree, built here and run on the host below
  cmake -S $S -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I$ENV/include > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; echo "[build] TEST CONFIGURE FAILED"; exit 1; }
  cmake --build $SCR/t -j ${SLURM_CPUS_PER_TASK:-24} -- -k > $OUTD/test-build.log 2>&1 || echo "[build] some tests did not build" >> $OUTD/test-build.log
fi
EOS
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh
if [ "$TESTS" != 0 ]; then
  tail -2 $OUTD/test-build.log
  ( export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib; echo "host glibc: $(ldd --version | head -1)" > $OUTD/ctest-full.txt
    cd $SCR/t && PZ2_TEST_MAX_MB=8 ctest -j ${SLURM_CPUS_PER_TASK:-24} --output-on-failure >> $OUTD/ctest-full.txt 2>&1; echo "ctest exit $?" >> $OUTD/ctest-full.txt )
  grep -E "tests passed|\*\*\*Failed|ctest exit" $OUTD/ctest-full.txt | tail -30 || true
fi
[ "$TESTS" = only ] && { echo "[build] $NAME tests only, $(( $(date +%s) - T0 )) s"; exit 0; }
echo "$COMMIT" > $OUTD/commit-gpu
SHA=$(sha256sum $OUTD/singlet-gpu | cut -d' ' -f1); EMB=$(cat $SCR/archs.txt 2>/dev/null | sed 's/ *$//')
{ echo "base_commit=$COMMIT"; echo "patch_sha256=$(sha256sum $OUTD/patch.diff | cut -d' ' -f1)"; echo "name=$NAME"; echo "recipe=jobs/gpu-runtime/rt_build.sbatch"
  echo "cuda_archs_requested=${CUDA_ARCH:-70;75;90}"; echo "cuda_archs_embedded=$EMB"; echo "nvcc=$(cat $SCR/nvcc.txt 2>/dev/null)"
  echo "sha256=$SHA"; echo "built_utc=$(date -u +%FT%TZ)"; echo "slurm_job=$SLURM_JOB_ID"; } > $OUTD/BUILD.txt
cat $OUTD/BUILD.txt
ldd $OUTD/singlet-gpu | grep -iE "cuda|not found" || true
echo "[build] $NAME ok in $(( $(date +%s) - T0 )) s"
