#!/bin/bash
#SBATCH --job-name=s8-ctestp
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile and run pileup C++ tests (TESTS, space separated) (links htslib; samtools from the atacbench env) from a branch (BRANCH, TEST). No CMake build of the tree.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git; BR=${BRANCH:?}; TESTS=${TESTS:-${TEST:?}}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/s8/ctestp
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/s8/ctestp | tar -x -C $SCR/src
echo "commit $(git -C $BARE rev-parse refs/s8/ctestp)"
export PATH=$ENV/bin:$W/envs/atacbench/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
cd $SCR/src; RC=0
for T in $TESTS; do
  echo "=== $T"
  g++ -std=c++17 -O1 -g -fopenmp -msse4.2 -Iinclude -Iinclude/singlet/pileup -Iinclude/singlet -I$ENV/include tests/cpp/$T.cpp -o $SCR/t_$T -L$ENV/lib -Wl,-rpath,$ENV/lib -lhts -lzstd -lz -lbz2 -llzma -lpthread || { echo "BUILD FAILED $T"; RC=1; continue; }
  $SCR/t_$T > $SCR/$T.out 2>&1; rc=$?; grep -a -E "passed|failed|FAIL|SKIP|ok" $SCR/$T.out | tail -6 || tail -3 $SCR/$T.out; echo "$T rc=$rc"; [ $rc = 0 ] || RC=1
done
exit $RC
