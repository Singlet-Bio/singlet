#!/bin/bash
#SBATCH -J bt-labels
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sbatch --export=ALL,PIN=<sha>,BRANCH=<branch>,TESTS="test_a test_b" build_test.sbatch : build and run the named ctest executables of a branch commit (CPU, pbench toolchain)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/${BRANCH}:refs/rg/btbr
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $PIN | tar -x -C $SCR/src
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5 LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $SCR/src -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=ON > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
for t in $TESTS; do cmake --build $SCR/b --target $t -j 8 > $SCR/b_$t.log 2>&1 || { grep -m20 -B2 -A6 "error" $SCR/b_$t.log || tail -30 $SCR/b_$t.log; exit 1; }; done
cd $SCR/b; for t in $TESTS; do f=$(find . -name $t -type f | head -1); echo "== $t"; $f 2>&1 | tail -25; done
