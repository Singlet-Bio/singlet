#!/bin/bash
#SBATCH --job-name=e15-starbuild
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E15 (docs/gpu-engine/fable/E15-posterior.md): `e15star`, a plain STAR from this branch's vendored STAR (src/star; CPU only, the GPU engine
# off, so SeqCacheGpuPrefill.cpp compiles to its stubs), main in bench/fable/posterior/star_main.cpp; recipe of bench/gpu/jobs/stardump_build.sbatch.
# The target is added to a scratch copy of src/pipeline/CMakeLists.txt only. Output: $OUTD/e15star (+ e15star.commit). Built on short-gpu
# without a GPU request.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/posterior}; OUTD=${OUTD:-$W/jobs/fable/posterior/bin}; mkdir -p $OUTD
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/posterior-star
COMMIT=$(git -C $BARE rev-parse refs/fable/posterior-star); echo "COMMIT=$COMMIT"
SCR=/tmp/$USER-e15s-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf "${SCR:?}"' EXIT
git -C $BARE archive refs/fable/posterior-star | tar -x -C $SCR/src
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat >> $S/src/pipeline/CMakeLists.txt <<'EOF'
add_executable(e15star ${CMAKE_SOURCE_DIR}/bench/fable/posterior/star_main.cpp)
target_include_directories(e15star PRIVATE ${CMAKE_SOURCE_DIR}/src/star ${CMAKE_SOURCE_DIR}/include/singlet/star)
target_link_directories(e15star PRIVATE ${HTSLIB_LIBRARY_DIRS})
target_link_libraries(e15star PRIVATE singlet_star ${HTSLIB_LIBRARIES} ${ZSTD_LIBRARY} ZLIB::ZLIB OpenMP::OpenMP_CXX Threads::Threads)
EOF
T0=$(date +%s)
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target e15star -j ${SLURM_CPUS_PER_TASK:-16} > $SCR/b.log 2>&1 || { grep -m40 -B2 -A6 "rror" $SCR/b.log || tail -40 $SCR/b.log; exit 1; }
cp $SCR/b/src/pipeline/e15star $OUTD/e15star.new && mv $OUTD/e15star.new $OUTD/e15star
echo "$COMMIT" > $OUTD/e15star.commit
sha256sum $OUTD/e15star; ldd $OUTD/e15star | grep -iE "not found" || true
echo "[build] e15star ok in $(( $(date +%s) - T0 )) s"
