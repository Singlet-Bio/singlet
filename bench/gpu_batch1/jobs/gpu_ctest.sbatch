#!/bin/bash
#SBATCH --job-name=gb1-ctest
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU-enabled ctest. The ctest of bench/gpu_device/jobs/build.sbatch is a CPU test build, so test_gpu_pileup_labels is skipped there
# (tests/cpp/CMakeLists.txt). This job builds the tree with -DSINGLET_ENGINE_GPU=ON -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON
# with the toolchain recipe of build.sbatch, runs test_gpu_pileup_labels on the job's H100 and then the whole suite.
#
# Parameters (sbatch --export=ALL,NAME=value,...):
#   REV            required. A branch of the singlet bare clone or a full 40-digit commit.
#   EXPECT_COMMIT  required. The job stops unless REV resolves to a commit that starts with this (8 or more digits).
#   ONLY_GPU_TEST  1: build and run only test_gpu_pileup_labels (a quick check of the recipe, fits debug-gpu). Default 0.
#   EXTRA_CXX      extra C++ flags, as in build.sbatch.
# Results: $W/jobs/gpu-batch1/ctest-<commit8>/ with summary-<job>.txt, gpu_test-<job>.log and ctest-<job>.log. The log carries the same lines.
# Last lines of the log:
#   GPU TEST test_gpu_pileup_labels: PASS, FAIL or SKIP (with the reason)
#   CTEST SUITE: <passed> of <total> passed, <failed> failed, <skipped> skipped
# Exit status: 0 when the GPU test passed and nothing failed. 1 when something failed. 4 when the GPU test was skipped.
# debug-gpu: sbatch -p debug-gpu -c 8 --mem=64G --time=00:29:00 --export=ALL,REV=<commit>,EXPECT_COMMIT=<commit8>,ONLY_GPU_TEST=1 gpu_ctest.sbatch
set -euo pipefail
: "${REV:?REV: a branch or a full commit}" "${EXPECT_COMMIT:?EXPECT_COMMIT: the commit REV must resolve to}"
[ ${#EXPECT_COMMIT} -ge 8 ] || { echo "EXPECT_COMMIT needs 8 or more hex digits"; exit 2; }
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git
URL="$(git -C $BARE config remote.origin.url)"
SREF=refs/gb1/ctest-$SLURM_JOB_ID
SCR=/tmp/$USER-gb1ct-$SLURM_JOB_ID; mkdir -p $SCR/src
trap 'git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf $SCR' EXIT
# one fetch at a time (concurrent jobs share the bare clone), then check what was fetched
if [[ $REV =~ ^[0-9a-f]{40}$ ]]; then
  ( flock 9; git -C $BARE fetch -q "$URL" +$REV:$SREF ) 9>$BARE/.gn-fetch.lock
  WANT=$REV
else
  WANT=$(git ls-remote "$URL" refs/heads/$REV | cut -f1); [ -n "$WANT" ] || { echo "no branch $REV"; exit 3; }
  ( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$REV:$SREF ) 9>$BARE/.gn-fetch.lock
fi
COMMIT=$(git -C $BARE rev-parse $SREF^{commit}); [ "$COMMIT" = "$WANT" ] || { echo "fetched $COMMIT, asked for $WANT"; exit 3; }
case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: $REV is $COMMIT, EXPECT_COMMIT is $EXPECT_COMMIT"; exit 2 ;; esac
OUTD=$W/jobs/gpu-batch1/ctest-${COMMIT:0:8}; mkdir -p $OUTD
SUM=$OUTD/summary-$SLURM_JOB_ID.txt
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) tree=$COMMIT engine_gpu=ON tests=ON only_gpu_test=${ONLY_GPU_TEST:-0} $(date -Is)" | tee $SUM
git -C $BARE archive $SREF | tar -x -C $SCR/src
S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
if [ "${ONLY_GPU_TEST:-0}" = 1 ]; then TARGETS="--target test_gpu_pileup_labels"; else TARGETS=""; fi
cat > $SCR/run.sh <<EOS
set -uo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
# the test targets need zstd.h from the env (build.sbatch's CPU test build of c92a494b failed on it: job 800221, 31 of 156 tests not built)
export CPLUS_INCLUDE_PATH=$ENV/include C_INCLUDE_PATH=$ENV/include LIBRARY_PATH=$ENV/lib
\$CXX --version | head -1; nvcc --version | tail -1; nvidia-smi -L
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=ON \
  -DCMAKE_CXX_FLAGS="${EXTRA_CXX:-}" -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="90" > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; echo "CONFIGURE FAILED"; exit 1; }
grep -E "GPU engine:" $SCR/c.log
cmake --build $SCR/b -j ${SLURM_CPUS_PER_TASK:-24} $TARGETS -- -k > $SCR/b.log 2>&1 || { grep -m40 -B2 -A8 "rror" $SCR/b.log | tail -60; echo "[build] SOME TARGETS FAILED TO BUILD"; }
T=\$(find $SCR/b -name test_gpu_pileup_labels -type f | head -1)
if [ -z "\$T" ]; then echo "GPU TEST test_gpu_pileup_labels: FAIL (not built)"; exit 1; fi
cd \$(dirname \$T)
./test_gpu_pileup_labels > $OUTD/gpu_test-$SLURM_JOB_ID.log 2>&1; rc=\$?
tail -5 $OUTD/gpu_test-$SLURM_JOB_ID.log
case \$rc in
  0)  echo "GPU TEST test_gpu_pileup_labels: PASS" ;;
  77) echo "GPU TEST test_gpu_pileup_labels: SKIP (\$(grep -m1 SKIP $OUTD/gpu_test-$SLURM_JOB_ID.log))" ;;
  *)  echo "GPU TEST test_gpu_pileup_labels: FAIL (exit \$rc)" ;;
esac
if [ "${ONLY_GPU_TEST:-0}" = 1 ]; then [ \$rc = 77 ] && exit 4; exit \$rc; fi
cd $SCR/b
PZ2_TEST_MAX_MB=8 ctest -j ${SLURM_CPUS_PER_TASK:-24} --output-on-failure --timeout 3600 > $OUTD/ctest-$SLURM_JOB_ID.log 2>&1
tail -25 $OUTD/ctest-$SLURM_JOB_ID.log
grep -E "Test +#[0-9]+: .*(Failed|Skipped|Not Run|Timeout|Exception)" $OUTD/ctest-$SLURM_JOB_ID.log || true
grep -E "test_gpu_pileup_labels" $OUTD/ctest-$SLURM_JOB_ID.log | head -3
TOT=\$(grep -cE "Test +#[0-9]+:" $OUTD/ctest-$SLURM_JOB_ID.log); PASS=\$(grep -cE "Test +#[0-9]+: .*Passed" $OUTD/ctest-$SLURM_JOB_ID.log)
SKIP=\$(grep -cE "Test +#[0-9]+: .*Skipped" $OUTD/ctest-$SLURM_JOB_ID.log); FAILN=\$((TOT - PASS - SKIP))
echo "CTEST SUITE: \$PASS of \$TOT passed, \$FAILN failed, \$SKIP skipped"
if [ \$rc = 0 ] && [ \$FAILN = 0 ]; then exit 0; elif [ \$rc = 77 ] && [ \$FAILN = 0 ]; then exit 4; else exit 1; fi
EOS
T0=$(date +%s)
set +e
apptainer exec --nv -B $W -B $SCR $IMG bash $SCR/run.sh 2>&1 | tee -a $SUM
RC=${PIPESTATUS[0]}
set -e
echo "[gpu_ctest] ${COMMIT:0:8} exit $RC in $(( $(date +%s) - T0 )) s" | tee -a $SUM
exit $RC
