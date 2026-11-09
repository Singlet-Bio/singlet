#!/bin/bash
#SBATCH --job-name=detect-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU build of a branch of the engine on a short-cpu node, plus the host-side protocol unit tests (ctest -R protocol).
# BRANCH is fetched by URL into the bare clone. Output: $W/jobs/detect/bin/singlet-cpu-<sha8> (and a copy named singlet-cpu).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
BR=${BRANCH:?BRANCH}; OUTD=$W/jobs/detect/bin; mkdir -p $OUTD
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/detect/build
COMMIT=$(git -C $BARE rev-parse refs/detect/build); S8=${COMMIT:0:8}; echo "COMMIT=$COMMIT"
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-detect-build-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/detect/build | tar -x -C $SCR/src
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
cmake -S $SCR/src -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/cmake.log 2>&1 || { tail -30 $SCR/cmake.log; exit 1; }
cmake --build $SCR/b --target singlet test_protocol_route test_protocol_forced_only test_protocol_short_barcode_read test_protocol_untrimmed_r1 test_ws2_protocol_hardening test_protocol_confidence_override -j 8 > $SCR/build.log 2>&1 || { grep -m20 -B2 -A6 "error" $SCR/build.log || tail -40 $SCR/build.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $OUTD/singlet-cpu-$S8 && cp $OUTD/singlet-cpu-$S8 $OUTD/singlet-cpu; echo $COMMIT > $OUTD/commit
cd $SCR/b && ctest -R "protocol" --output-on-failure 2>&1 | tail -25
echo "[build] ok $S8"
