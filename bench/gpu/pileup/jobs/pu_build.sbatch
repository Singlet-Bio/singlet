#!/bin/bash
#SBATCH --job-name=w11-pu-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 pileup: non-PGO dev build of prod/campaign (c4a1be38) with bench/gpu/pileup/pileup_dump.patch -> $W/jobs/w11/pileup/bin/singlet-pudump.
# The shared worktree src/singlet-py/prod__campaign is NOT touched (source comes from `git archive` of a private ref of the bare repo).
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; M=$W/jobs/w11/pileup
BARE=$W/repos/singlet-py.git; BR=prod/campaign
git -C $BARE fetch -q "$(git -C $BARE remote get-url origin)" +refs/heads/$BR:refs/w11/campaign-pileup || echo "[build] fetch failed, using the local ref"
REF=refs/w11/campaign-pileup; git -C $BARE rev-parse -q --verify $REF >/dev/null || REF=refs/heads/$BR
COMMIT=$(git -C $BARE rev-parse $REF); echo "COMMIT=$COMMIT"
case $COMMIT in c4a1be38*) ;; *) echo "[build] expected c4a1be38"; exit 1;; esac
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $REF | tar -x -C $SCR/src
cd $SCR/src
patch -p1 --no-backup-if-mismatch < $M/pileup_dump.patch
grep -rl --include=CMakeLists.txt -e "-march=native" . | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' src/star/bam_cat.c
export TMPDIR=$SCR
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/cmake.log 2>&1 || { tail -30 $SCR/cmake.log; exit 1; }
cmake --build $SCR/b --target singlet -j 8 > $SCR/build.log 2>&1 || { grep -m40 -B2 -A6 "rror" $SCR/build.log; tail -20 $SCR/build.log; exit 1; }
mkdir -p $M/bin; cp $SCR/b/src/pipeline/singlet $M/bin/singlet-pudump.new && mv $M/bin/singlet-pudump.new $M/bin/singlet-pudump
sha256sum $M/bin/singlet-pudump; echo "$COMMIT" > $M/bin/singlet-pudump.commit
echo "[build] done"
