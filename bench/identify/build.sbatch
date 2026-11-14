#!/bin/bash
#SBATCH --job-name=identify-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build singlet identify on Clipper from a fresh shallow clone of the branch (no shared worktrees are touched).
#   sbatch build.sbatch bin            pipeline binary -> jobs/identify/bin/singlet-<sha8>
#   sbatch build.sbatch test           test_identify only, then run it
#   sbatch build.sbatch ctest          every C++ test target, then ctest
# BRANCH (default feat/identify) selects the branch.
set -eu
WHAT=${1:-bin}
BR=${BRANCH:-feat/identify}
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; J=$W/jobs/identify
mkdir -p $J/bin
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git clone -q --depth 1 --branch $BR git@github.com:Singlet-Bio/singlet.git $SCR/src
SHA=$(git -C $SCR/src rev-parse --short=8 HEAD); git -C $SCR/src log --oneline -1
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
case $WHAT in
  bin)
    cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$SHA > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
    cmake --build $SCR/b --target singlet -j ${SLURM_CPUS_PER_TASK} > $SCR/b.log 2>&1 || { grep -m40 -B2 -A8 "error" $SCR/b.log; exit 1; }
    mkdir -p $J/wl; cp -n $SCR/src/python/singlet/data/whitelists/* $J/wl/; for f in 3M-3pgex-may-2023.txt; do [ -e $J/wl/$f ] || ln -s $W/bpub/whitelists/$f $J/wl/$f; done; cp $SCR/b/src/pipeline/singlet $J/bin/singlet-$SHA; ln -sfn singlet-$SHA $J/bin/singlet; ls -la $J/bin/ ;;
  test)
    cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF -DCMAKE_BUILD_TYPE=Release > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
    cmake --build $SCR/t --target test_identify -j ${SLURM_CPUS_PER_TASK} > $SCR/tb.log 2>&1 || { grep -m40 -B2 -A8 "error" $SCR/tb.log; exit 1; }
    (cd $SCR/t/tests/cpp && ./test_identify) ;;
  ctest)
    cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF -DCMAKE_BUILD_TYPE=Release > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
    cmake --build $SCR/t -j ${SLURM_CPUS_PER_TASK} -- -k > $SCR/tb.log 2>&1 || { grep -c "error" $SCR/tb.log; grep -m20 "Error " $SCR/tb.log; }
    (cd $SCR/t && ctest -j 6 --timeout 900 --output-on-failure > $J/ctest-$SHA.log 2>&1 || true; tail -25 $J/ctest-$SHA.log) ;;
esac
