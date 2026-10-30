#!/bin/bash
#SBATCH --job-name=w11-m6-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 M6: non-PGO dev build of prod/campaign with bench/gpu/m6/tuple_dump.patch -> $W/jobs/w11/m6/bin/singlet-m6dump.
# The shared worktree src/singlet-py/prod__campaign is NOT touched (source comes from `git archive` of the bare repo's ref).
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; M=$W/jobs/w11/m6
BARE=$W/repos/singlet-py.git; BR=prod/campaign
# private ref only: refs/heads/prod/campaign is checked out by the campaign worktree and must not be touched
git -C $BARE fetch -q origin +refs/heads/$BR:refs/w11/campaign-m6
COMMIT=$(git -C $BARE rev-parse refs/w11/campaign-m6); echo "COMMIT=$COMMIT"
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/w11/campaign-m6 | tar -x -C $SCR/src
cd $SCR/src
patch -p1 --no-backup-if-mismatch < $M/tuple_dump.patch
grep -rl --include=CMakeLists.txt -e "-march=native" . | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' src/star/bam_cat.c
export TMPDIR=$SCR
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/cmake.log 2>&1 || { tail -30 $SCR/cmake.log; exit 1; }
cmake --build $SCR/b --target singlet -j 16 > $SCR/build.log 2>&1 || { grep -m40 -B2 -A6 "rror" $SCR/build.log; tail -20 $SCR/build.log; exit 1; }
mkdir -p $M/bin; cp $SCR/b/src/pipeline/singlet $M/bin/singlet-m6dump.new && mv $M/bin/singlet-m6dump.new $M/bin/singlet-m6dump
sha256sum $M/bin/singlet-m6dump; echo "$COMMIT" > $M/bin/singlet-m6dump.commit
ldd $M/bin/singlet-m6dump | head -30
echo "[build] done"
