#!/bin/bash
#SBATCH --job-name=bb-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch build.sbatch <branch> <bin-name> [test ...]
# Fetch <branch>, build the pipeline binary as pipeline-benchmark/bin/<bin-name> (x86-64-v3, as bench/prod_atlas/build.sbatch),
# then build and run the listed C++ tests (names without path, e.g. test_umi_store_scale).
set -euo pipefail
BR=$1; NAME=$2; shift 2; TESTS=("$@")
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; SRC=$W/src/singlet-py/bb__${BR//\//__}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q --detach $SRC refs/heads/$BR
git -C $SRC reset -q --hard refs/heads/$BR; COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
if [ "$NAME" != "-" ]; then
  cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
  cmake --build $SCR/b --target singlet -j 8 > $SCR/b.log 2>&1 || { grep -m30 -B2 -A8 "error" $SCR/b.log; exit 1; }
  cp $SCR/b/src/pipeline/singlet $PB/bin/$NAME
  sha256sum $PB/bin/$NAME
fi
if [ ${#TESTS[@]} -gt 0 ]; then
  cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF -DCMAKE_BUILD_TYPE=Release > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
  for t in "${TESTS[@]}"; do
    cmake --build $SCR/t --target $t -j 8 > $SCR/tb.log 2>&1 || { grep -m30 -A8 error $SCR/tb.log; echo "BUILD FAIL $t"; continue; }
    echo "== $t"; (cd $SCR/t/tests/cpp && ./$t | grep -v "PASS" | tail -15; echo "exit ${PIPESTATUS[0]}")
  done
fi
