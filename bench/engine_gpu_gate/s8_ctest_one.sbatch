#!/bin/bash
#SBATCH --job-name=s8-ctest1
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile and run one header-only C++ test from a branch (BRANCH, TEST): g++ from the pbench env, include/ on the path, zstd linked. No CMake build of the tree.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git; BR=${BRANCH:?}; TEST=${TEST:?}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/s8/ctest
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/s8/ctest | tar -x -C $SCR/src
export PATH=$ENV/bin:$PATH
cd $SCR/src && g++ -std=c++17 -O1 -g -fopenmp -msse4.2 -Iinclude -I$ENV/include tests/cpp/$TEST.cpp -o $SCR/t -L$ENV/lib -Wl,-rpath,$ENV/lib -lzstd -lpthread
$SCR/t
