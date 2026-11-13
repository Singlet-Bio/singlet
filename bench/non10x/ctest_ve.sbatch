#!/bin/bash
#SBATCH --job-name=vn-ctest
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# validation-engine copy (D-152) of jobs/gpu-pr2/ctest-host2.sbatch (output under results/validation-engine/non10x/ctest). Original header: PR-2 gate G1: the CPU unit tests of a branch, built in the container (conda toolchain) and RUN ON THE HOST.
# Why: the first test builds ran ctest inside the Ubuntu 22.04 container. The conda toolchain links 27 of the test binaries against libc symbols of
# GLIBC_2.38, which the container's libc (2.35) lacks, so they did not start. The cluster hosts run the same binaries (the gate jobs run singlet-gpu on the host).
# Second fix: the four test_pz2_* tests include zstd.h, which the container lacks, so the test build gets -I$ENV/include (the conda zstd).
# Needs BR (branch) and NAME (output dir name). Output: $W/jobs/gpu-pr2/ctest/$NAME/{ctest-full.txt,ctest-LastTest.log,test-build.log,commit}. No singlet binary is built.
set -uo pipefail
: "${BR:?}" "${NAME:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; OUTD=$W/results/validation-engine/non10x/ctest/$NAME; [ -e $OUTD/ctest-full.txt ] && { echo "refusing to overwrite $OUTD"; exit 2; }; mkdir -p $OUTD
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$BR | cut -f1); [ -n "$WANT" ] || { echo "no branch $BR"; exit 3; }
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/gd/ct-$NAME ) 9>$BARE/.gn-fetch.lock
COMMIT=$(git -C $BARE rev-parse refs/gd/ct-$NAME); echo "COMMIT=$COMMIT NAME=$NAME"; [ "$COMMIT" = "$WANT" ] || { echo "fetched $COMMIT, branch tip is $WANT"; exit 3; }
echo $COMMIT > $OUTD/commit
SCR=/tmp/$USER-ct-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/gd/ct-$NAME | tar -x -C $SCR/src
S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
cat > $SCR/build.sh <<EOS
set -uo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-I$ENV/include > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
cmake --build $SCR/t -j ${SLURM_CPUS_PER_TASK:-24} -- -k > $OUTD/test-build.log 2>&1; echo "build exit \$?" >> $OUTD/test-build.log
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh
tail -3 $OUTD/test-build.log
# run on the host, with the conda toolchain's libraries
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
echo "host glibc: $(ldd --version | head -1)" | tee $OUTD/ctest-full.txt
( cd $SCR/t && PZ2_TEST_MAX_MB=8 ctest -j ${SLURM_CPUS_PER_TASK:-24} --output-on-failure >> $OUTD/ctest-full.txt 2>&1; echo "ctest exit $?" >> $OUTD/ctest-full.txt; cp Testing/Temporary/LastTest.log $OUTD/ctest-LastTest.log 2>/dev/null || true )
tail -40 $OUTD/ctest-full.txt | tee $OUTD/ctest.txt
# a failing ctest fails the job (ctest's own exit status is recorded in ctest-full.txt)
grep -aq "^ctest exit 0$" $OUTD/ctest-full.txt || { echo "CTEST FAILED"; exit 1; }
