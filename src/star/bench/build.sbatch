#!/bin/bash
#SBATCH --job-name=sa-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch build.sbatch 'tag1=FLAGS' 'tag2=FLAGS' ...
# Build perf/star-accel HEAD (x86-64-v3) once per variant; only STAR objects differ between
# variants (SINGLET_STAR_EXTRA_FLAGS). Installs bin/singlet-accel-<tag>.
# Env: CMAKE_EXTRA (extra cmake args), BR (branch, default perf/star-accel).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BR=${BR:-perf/star-accel}
git -C $W/repos/singlet-py.git fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
SRC=$W/src/singlet-py/${BR//\//__}
[ -d $SRC ] || git -C $W/repos/singlet-py.git worktree add -f $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_EXTRA:-} > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
for spec in "$@"; do
  tag=${spec%%=*}; flags=${spec#*=}
  cmake -S $SCR/src -B $SCR/b "-DSINGLET_STAR_EXTRA_FLAGS=$flags" > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
  t0=$(date +%s)
  cmake --build $SCR/b --target singlet -j 16 > $SCR/b.log 2>&1 || { grep -m20 -B2 -A8 "error" $SCR/b.log; exit 1; }
  cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-accel-$tag
  echo "built singlet-accel-$tag flags='$flags' in $(( $(date +%s)-t0 ))s"
done
