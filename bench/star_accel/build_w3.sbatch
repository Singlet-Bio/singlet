#!/bin/bash
#SBATCH --job-name=w3-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch build_w3.sbatch [branch=perf/w3-speedups] [name=singlet-w3]
# Builds <branch> twice, same recipe as jobs/campaign/build_campaign.sbatch (bin/singlet-campaign, x86-64-v3):
#   $PB/bin/<name>        PGO, STAR trained on a 2M-read 3' v3 sample (pbmc2m) and a 2M-read 5' sample (pbmc5p_v1)
#   $PB/bin/<name>-nopgo  the same sources without PGO
# then builds and runs the C++ tests.
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BR=${1:-perf/w3-speedups}; NAME=${2:-singlet-w3}
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
REF=/mnt/projects/debruinz_project/cellarium/reference
# Training runs see the production whitelists, so STARsolo takes the CB_UMI_Simple whitelist path as in the campaign.
export TMPDIR=$SCR CMAKE_ARGS="-DSINGLET_GIT_COMMIT=$COMMIT" SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF

echo "[build] non-PGO"
cmake -S $SCR/src -B $SCR/n -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release $CMAKE_ARGS > $SCR/n.cmake.log 2>&1
cmake --build $SCR/n --target singlet -j 16 > $SCR/n.build.log 2>&1 || { grep -m40 -B2 -A8 "rror" $SCR/n.build.log; exit 1; }
cp $SCR/n/src/pipeline/singlet $PB/bin/$NAME-nopgo.new && mv $PB/bin/$NAME-nopgo.new $PB/bin/$NAME-nopgo

echo "[build] 5' training sample (first 2M read pairs of pbmc5p_v1, encoded with bin/singlet-campaign)"
D5=$PB/data/pbmc5p_v1/sub
for r in R1 R2; do zcat $D5/sub_S1_L001_${r}_001.fastq.gz | head -n 8000000 > $SCR/t5_$r.fastq || true; done
$PB/bin/singlet-campaign encode --reads $SCR/t5_R1.fastq $SCR/t5_R2.fastq -o $SCR/train5p.1fq > $SCR/enc5p.log 2>&1 \
  || { tail -20 $SCR/enc5p.log; exit 1; }
rm -f $SCR/t5_R1.fastq $SCR/t5_R2.fastq
grep -E "Done|Protocol" $SCR/enc5p.log
cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train3p.1fq

echo "[build] PGO"
bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train3p.1fq:$SCR/train5p.1fq $REF/GRCh38-2024-A/star_2.7.11b \
  $REF/GRCh38-2024-A/genes/genes.gtf $REF "" || { for f in $SCR/b.*.log; do echo == $f; tail -n 25 $f; done; exit 1; }
grep -E "protocol_id|\[singlet\] Total" $SCR/b.train.log
cp $SCR/b/src/pipeline/singlet $PB/bin/$NAME.new && mv $PB/bin/$NAME.new $PB/bin/$NAME
sha256sum $PB/bin/$NAME $PB/bin/$NAME-nopgo

echo "[build] tests"
export CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1 || { tail -30 $SCR/tc.log; exit 1; }
cmake --build $SCR/t -j 16 -- -k > $SCR/tb.log 2>&1 || { grep -m60 -B2 -A8 "rror" $SCR/tb.log || tail -40 $SCR/tb.log; echo "[build] TEST BUILD FAILED"; }
cd $SCR/t && PZ2_TEST_MAX_MB=8 ctest -j 16 --output-on-failure 2>&1 | tail -40 || true
