#!/bin/bash
#SBATCH --job-name=dbl-ctest
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Host ctest of a branch against its base, same recipe, whole node (CPU build with tests, no CUDA). The failing sets are diffed:
# failures present on the base too are environment, not the change.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
NEW=fix/doublet-expression-score; BASE=gpu/frontier; OUT=$W/jobs/dblfix/ctest; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; NC=$(nproc)
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
one() {  # label branch
  L=$1; BR=$2
  git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/dblct/$L
  COMMIT=$(git -C $BARE rev-parse refs/dblct/$L); echo "$L $COMMIT" | tee -a $OUT/commits.txt
  mkdir -p $SCR/src-$L; git -C $BARE archive $COMMIT | tar -x -C $SCR/src-$L; S=$SCR/src-$L
  grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
  grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
  cat > $SCR/t-$L.sh <<EOS
set -uo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export CPLUS_INCLUDE_PATH=$ENV/include C_INCLUDE_PATH=$ENV/include LIBRARY_PATH=$ENV/lib
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b-$L -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON -DSINGLET_ENGINE_GPU=OFF > $SCR/c-$L.log 2>&1 || { tail -30 $SCR/c-$L.log; exit 1; }
cmake --build $SCR/b-$L -j $NC -- -k > $OUT/build-$L.log 2>&1
cd $SCR/b-$L && ctest -j 16 --timeout 900 > $OUT/ctest-$L.log 2>&1
EOS
  apptainer exec -B $W -B $SCR $IMG bash $SCR/t-$L.sh
  grep -E "tests passed" $OUT/ctest-$L.log
  sed -n '/The following tests FAILED/,$p' $OUT/ctest-$L.log | grep -oE "[0-9]+ - [a-z_0-9]+" | sed 's/^[0-9]* - //' | sort > $OUT/failed-$L.txt
  rm -rf $S $SCR/b-$L
}
one base $BASE
one new $NEW
echo "== failing on new only:"; comm -13 $OUT/failed-base.txt $OUT/failed-new.txt
echo "== failing on base only:"; comm -23 $OUT/failed-base.txt $OUT/failed-new.txt
echo "== failing on both: $(comm -12 $OUT/failed-base.txt $OUT/failed-new.txt | wc -l)"
