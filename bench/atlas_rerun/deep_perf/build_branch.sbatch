#!/bin/bash
#SBATCH --job-name=dp-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch build_branch.sbatch <branch> <outname> [pgo]
# Build <branch> of Singlet-Bio/singlet -> pipeline-benchmark/bin/<outname> (x86-64-v3; PGO like singlet-campaign
# when the 3rd argument is "pgo").
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BR=$1; OUTNAME=$2; MODE=${3:-plain}
BARE=$W/repos/singlet-py.git; SRC=$W/src/singlet-py/${BR//\//__}
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
REF=/mnt/projects/debruinz_project/cellarium/reference
if [ "$MODE" = pgo ]; then
  cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train.1fq
  export CMAKE_ARGS="-DSINGLET_GIT_COMMIT=$COMMIT"
  bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train.1fq $REF/GRCh38-2024-A/star_2.7.11b \
    $REF/GRCh38-2024-A/genes/genes.gtf $REF "" || { for f in $SCR/b.*.log; do echo == $f; tail -n 25 $f; done; exit 1; }
else
  cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
  cmake --build $SCR/b --target singlet -j 16 > $SCR/bb.log 2>&1 || { grep -m40 -B2 -A6 "error" $SCR/bb.log || tail -40 $SCR/bb.log; exit 1; }
fi
cp $SCR/b/src/pipeline/singlet $PB/bin/$OUTNAME.new && mv $PB/bin/$OUTNAME.new $PB/bin/$OUTNAME
sha256sum $PB/bin/$OUTNAME
if [ -n "${TESTS:-}" ]; then
  cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
  for t in $TESTS; do
    cmake --build $SCR/t --target $t -j 16 > $SCR/tb.log 2>&1 || { grep -m40 -B2 -A6 "error" $SCR/tb.log; exit 1; }
    f=$(find $SCR/t -name $t -type f -perm -u+x | head -1); echo "== $t"; $f 2>&1 | grep -v PASS | tail -20
  done
fi
