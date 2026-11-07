#!/bin/bash
#SBATCH --job-name=fqrs-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch [--export=ALL,PGO=1,TAG=name] build.sbatch
# Builds perf/fq-read-source -> $W/jobs/fq-read-source/bin/singlet-fqrs[-TAG]. PGO=1: the campaign recipe (pgo_build.sh,
# x86-64-v3, pbmc2m training), comparable in speed with pipeline-benchmark/bin/singlet-campaign; otherwise a plain
# Release build for compile checks. Then the C++ unit tests (test_read_source among them).
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; BR=perf/fq-read-source; SRC=$W/src/singlet-py/perf__fq-read-source
OUT=$W/jobs/fq-read-source/bin; mkdir -p $OUT; TAG=${TAG:-dev}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q -f $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
REF=/mnt/projects/debruinz_project/cellarium/reference
export TMPDIR=$SCR
if [ "${PGO:-0}" = 1 ]; then
  cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train.1fq
  export CMAKE_ARGS="-DSINGLET_GIT_COMMIT=$COMMIT"
  bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train.1fq $REF/GRCh38-2024-A/star_2.7.11b \
    $REF/GRCh38-2024-A/genes/genes.gtf $REF "" || { for f in $SCR/b.*.log; do echo == $f; tail -n 25 $f; done; exit 1; }
else
  cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
  cmake --build $SCR/b -j 16 --target singlet > $SCR/m.log 2>&1 || { grep -m40 -B3 -A12 "error" $SCR/m.log || tail -60 $SCR/m.log; exit 1; }
fi
cp $SCR/b/src/pipeline/singlet $OUT/singlet-fqrs-$TAG.new && mv $OUT/singlet-fqrs-$TAG.new $OUT/singlet-fqrs-$TAG
echo "$COMMIT" > $OUT/singlet-fqrs-$TAG.commit; sha256sum $OUT/singlet-fqrs-$TAG
echo "[build] tests"
export CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
cmake --build $SCR/t -j 16 -- -k > $SCR/tb.log 2>&1 || { grep -m60 -B2 -A8 "rror" $SCR/tb.log || tail -40 $SCR/tb.log; echo "[build] TEST BUILD FAILED"; }
cd $SCR/t && PZ2_TEST_MAX_MB=8 ctest -j 16 --output-on-failure 2>&1 | tail -60 || true
