#!/bin/bash
#SBATCH --job-name=w11-sdbuild
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build `stardump`: the vendored STAR of prod/campaign with ReadAlign_maxMappableLength2strands.cpp replaced by the dumping
# copy and ReadAlign_oneRead.cpp by the M5 truth copy (src/gpu/stardump/). Scratch copy only; src/star in git is untouched. Output: $W/jobs/w11/bin/stardump.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench
BARE=$W/repos/singlet-py.git; BR=prod/campaign; SD=$W/jobs/w11/stardump
# fetch by URL (not the remote name) so the remote's mirror refspec does not try to move the checked-out prod/campaign
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
# swap in the dumping copy (same file name, so STAR's glob compiles it instead of the original)
cp $SD/ReadAlign_maxMappableLength2strands_dump.cpp $S/src/star/ReadAlign_maxMappableLength2strands.cpp
cp $SD/dump_format.h $S/src/star/dump_format.h
cp $SD/ReadAlign_oneRead_dump.cpp $S/src/star/ReadAlign_oneRead.cpp
cp $SD/m5_dump.h $S/src/star/m5_dump.h
mkdir -p $S/w11 && cp $SD/stardump_main.cpp $S/w11/
cat >> $S/src/pipeline/CMakeLists.txt <<'EOF'
add_executable(stardump ${CMAKE_SOURCE_DIR}/w11/stardump_main.cpp)
target_include_directories(stardump PRIVATE ${CMAKE_SOURCE_DIR}/src/star ${CMAKE_SOURCE_DIR}/include/singlet/star)
target_link_directories(stardump PRIVATE ${HTSLIB_LIBRARY_DIRS})
target_link_libraries(stardump PRIVATE singlet_star ${HTSLIB_LIBRARIES} ZLIB::ZLIB OpenMP::OpenMP_CXX Threads::Threads)
EOF
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target stardump -j 16 > $SCR/b.log 2>&1 || { grep -m40 -B2 -A6 "rror" $SCR/b.log || tail -40 $SCR/b.log; exit 1; }
mkdir -p $W/jobs/w11/bin
cp $SCR/b/src/pipeline/stardump $W/jobs/w11/bin/stardump.new && mv $W/jobs/w11/bin/stardump.new $W/jobs/w11/bin/stardump
echo "$COMMIT" > $W/jobs/w11/bin/stardump.commit
sha256sum $W/jobs/w11/bin/stardump; ldd $W/jobs/w11/bin/stardump | grep -v "=> /lib" | head
echo "[build] ok"
