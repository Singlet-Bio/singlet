#!/bin/bash
#SBATCH --job-name=if-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Non-PGO build of fix/intron-assignment -> pipeline-benchmark/bin/singlet-intronfix, then the molecule-label test.
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; BR=${BR:-fix/intron-rule-default}; SRC=$W/src/singlet-py/$(echo $BR | sed "s#/#__#g")
OUTNAME=${OUTNAME:-singlet-intronrule}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q -f $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
echo "COMMIT=$COMMIT"
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
export TMPDIR=$SCR CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target singlet -j 16 > $SCR/bb.log 2>&1 || { grep -m40 -B2 -A6 "error" $SCR/bb.log || tail -40 $SCR/bb.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $PB/bin/$OUTNAME.new && mv $PB/bin/$OUTNAME.new $PB/bin/$OUTNAME
sha256sum $PB/bin/$OUTNAME
echo "[build] test_molecule_labels"
cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
cmake --build $SCR/t --target test_molecule_labels -j 16 > $SCR/tb.log 2>&1 || { grep -m40 -B2 -A6 "error" $SCR/tb.log || tail -40 $SCR/tb.log; exit 1; }
f=$(find $SCR/t -name test_molecule_labels -type f -perm -u+x | head -1); $f | tail -40
if [ -n "${ALLTESTS:-}" ]; then
  cmake --build $SCR/t -j 16 -- -k > $SCR/tb2.log 2>&1 || { grep -m60 -B2 -A8 "rror" $SCR/tb2.log || tail -40 $SCR/tb2.log; echo "[build] TEST BUILD FAILED"; }
  cd $SCR/t && PZ2_TEST_MAX_MB=8 ctest -j 16 --output-on-failure 2>&1 | tail -60 || true
fi
