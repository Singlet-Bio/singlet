#!/bin/bash
#SBATCH --job-name=w11-m13-build
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 M13: (1) nvcc build of m13.cu (sm_90, -fmad=false) in the CUDA 12.4.1 devel image -> bin/m13;
# (2) non-PGO dev build of prod/campaign c4a1be38 + m13.patch -> $W/jobs/w11/m13/bin/singlet-m13 (never touches the campaign
# worktree or $W/pipeline-benchmark/bin).  Set SKIP_ENGINE=1 to rebuild only the CUDA program.
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; M=$W/jobs/w11/m13; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; BR=prod/campaign
mkdir -p $M/bin
# ---- CUDA program first (fast; a compile error should not wait for the engine build)
rm -f $M/bin/m13.new
apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -o $M/bin/m13.new $M/m13.cu 2>&1 | grep -v "Wno-deprecated" || true
[ -x $M/bin/m13.new ] && mv $M/bin/m13.new $M/bin/m13 && ls -la $M/bin/m13 || { echo "[build] nvcc failed"; exit 1; }
[ "${SKIP_ENGINE:-0}" = 1 ] && { echo "[build] CUDA only, done"; exit 0; }
# ---- engine
git -C $BARE fetch -q "$(git -C $BARE remote get-url origin)" +refs/heads/$BR:refs/w11/campaign-m13 || echo "[build] fetch failed, using the local ref"
REF=refs/w11/campaign-m13; git -C $BARE rev-parse -q --verify $REF >/dev/null || REF=refs/heads/$BR
COMMIT=$(git -C $BARE rev-parse $REF); echo "COMMIT=$COMMIT"
case $COMMIT in c4a1be38*) ;; *) echo "[build] expected c4a1be38"; exit 1;; esac
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $REF | tar -x -C $SCR/src
cd $SCR/src
patch -p1 --no-backup-if-mismatch < $M/m13.patch
grep -rl --include=CMakeLists.txt -e "-march=native" . | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' src/star/bam_cat.c
export TMPDIR=$SCR
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/cmake.log 2>&1 || { tail -30 $SCR/cmake.log; exit 1; }
cmake --build $SCR/b --target singlet -j 16 > $SCR/build.log 2>&1 || { grep -m40 -B2 -A6 "rror" $SCR/build.log; tail -20 $SCR/build.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $M/bin/singlet-m13.new && mv $M/bin/singlet-m13.new $M/bin/singlet-m13
sha256sum $M/bin/singlet-m13; echo "$COMMIT" > $M/bin/singlet-m13.commit
echo "[build] done"
