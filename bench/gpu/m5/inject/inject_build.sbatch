#!/bin/bash
#SBATCH --job-name=w11-injbuild
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build `singlet-m5inject`: the campaign engine (prod/campaign) with STAR's sequence cache preloadable from a file of GPU
# results (bench/gpu/m5/inject/apply_preload.py). Scratch copy only; src/star in git is untouched. Not PGO (outputs do not
# depend on it). Output: $W/jobs/w11/bin/singlet-m5inject.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
BARE=$W/repos/singlet-py.git; BR=prod/campaign; D=$W/jobs/w11/m5
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/w11/campaign
COMMIT=$(git -C $BARE rev-parse refs/w11/campaign); echo "COMMIT=$COMMIT"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/w11/campaign | tar -x -C $SCR/src
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
PY=$(ls $ENV/bin/python3 2>/dev/null || command -v python3); $PY $D/apply_preload.py $S/src/star/SeqCache.cpp
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target singlet -j 16 > $SCR/b.log 2>&1 || { grep -m40 -B2 -A6 "rror" $SCR/b.log || tail -40 $SCR/b.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $W/jobs/w11/bin/singlet-m5inject.new && mv $W/jobs/w11/bin/singlet-m5inject.new $W/jobs/w11/bin/singlet-m5inject
echo "$COMMIT" > $W/jobs/w11/bin/singlet-m5inject.commit
sha256sum $W/jobs/w11/bin/singlet-m5inject; echo "[build] ok"
