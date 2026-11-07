#!/bin/bash
#SBATCH --job-name=txf-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# PGO build of perf/txome-first -> pipeline-benchmark/bin/singlet-txf (x86-64-v3), same recipe as
# jobs/campaign/build_campaign.sbatch (which built bin/singlet-campaign from prod/campaign).
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; BR=perf/txome-first; SRC=$W/src/singlet-py/perf__txome-first
git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
[ -d $SRC ] || git -C $BARE worktree add -q -f $SRC $BR
git -C $SRC reset -q --hard refs/heads/$BR; COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
REF=/mnt/projects/debruinz_project/cellarium/reference
cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train.1fq
export TMPDIR=$SCR CMAKE_ARGS="-DSINGLET_GIT_COMMIT=$COMMIT"
bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train.1fq $REF/GRCh38-2024-A/star_2.7.11b \
  $REF/GRCh38-2024-A/genes/genes.gtf $REF "" || { for f in $SCR/b.*.log; do echo == $f; tail -n 25 $f; done; exit 1; }
cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-txf.new && mv $PB/bin/singlet-txf.new $PB/bin/singlet-txf
echo "COMMIT=$COMMIT"; sha256sum $PB/bin/singlet-txf
