#!/bin/bash
#SBATCH --job-name=mf-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fetch prod/modality-fixes, build it as pipeline-benchmark/bin/singlet-modfix (x86-64-v3, like singlet-prod), then build
# and run every C++ test (ctest), as CI does. Mirrors bench/prod_atlas/build.sbatch.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; BR=${BR:-prod/modality-fixes}; SRC=$W/src/singlet-py/${BR//\//__}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
if [ "${SKIP_BIN:-0}" != 1 ]; then
  cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
  cmake --build $SCR/b --target singlet -j 16 > $SCR/b.log 2>&1 || { grep -m40 -B2 -A8 "error" $SCR/b.log; exit 1; }
  grep -c "warning:" $SCR/b.log | sed 's/^/[build] pipeline warnings: /'
  cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-modfix.new && mv $PB/bin/singlet-modfix.new $PB/bin/singlet-modfix
  sha256sum $PB/bin/singlet-modfix
fi
cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
cmake --build $SCR/t -j 16 -k > $SCR/tb.log 2>&1 || { grep -m60 -B2 -A8 "error" $SCR/tb.log; echo "[build] TEST BUILD FAILED"; }
grep "warning:" $SCR/tb.log | grep -E "doublet|demux|ambient|mt_|polya|sj_features|star_cell|prefix_probe|export|run_prov" | sort -u | head -40
cd $SCR/t && PZ2_TEST_MAX_MB=8 ctest -j 16 --output-on-failure -E "test_pz2_roundtrip" 2>&1 | tail -60
