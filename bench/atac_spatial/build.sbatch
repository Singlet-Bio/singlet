#!/bin/bash
#SBATCH --job-name=as-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build the benchmark binary: this branch merged onto prod/modality-fixes (the atlas re-run engine), installed as
# pipeline-benchmark/bin/singlet-atacsp (x86-64-v3, like singlet-modfix). Then build and run the ATAC/Visium C++ tests.
# Mirrors jobs/modfix/build.sbatch.
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; BR=${BR:-claude/atac-spatial-benchmark-2ba7df}; BASE=${BASE:-prod/modality-fixes}
SRC=$W/src/singlet-py/${BR//\//__}
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR +refs/heads/$BASE:refs/heads/$BASE
[ -d $SRC ] || git -C $BARE worktree add -q --detach $SRC $BR
cd $SRC; git checkout -q --detach refs/heads/$BR; git reset -q --hard refs/heads/$BR
git -c user.name=bench -c user.email=bench@local merge -q --no-edit refs/heads/$BASE
COMMIT=$(git rev-parse HEAD); git log --oneline -3
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
if [ "${SKIP_BIN:-0}" != 1 ]; then
  cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
  cmake --build $SCR/b --target singlet -j 16 > $SCR/b.log 2>&1 || { grep -m40 -B2 -A8 "error" $SCR/b.log; exit 1; }
  grep "warning:" $SCR/b.log | grep -E "atac_|visium" | sort -u | head -30 || true
  cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-atacsp.new && mv $PB/bin/singlet-atacsp.new $PB/bin/singlet-atacsp
  sha256sum $PB/bin/singlet-atacsp
fi
cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
T="test_atac_fragment test_atac_peaks test_atac_qc test_atac_cell_caller test_atac_bin_counter test_atac_peak_caller test_atac_metrics test_visium_qc test_visium_spatial test_visium_tissue test_multiome_router"
for t in $T; do cmake --build $SCR/t --target $t -j 16 > $SCR/tb_$t.log 2>&1 || { grep -m30 -B2 -A8 "rror" $SCR/tb_$t.log | head -60; echo "[build] $t BUILD FAILED"; }; done
cd $SCR/t && ctest --output-on-failure -R "atac|visium|multiome" 2>&1 | tail -40 || true
