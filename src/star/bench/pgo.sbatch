#!/bin/bash
#SBATCH --job-name=sa-pgo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch pgo.sbatch TAG "STAR FLAGS" [CMAKE_ARGS]
# PGO build of perf/star-accel HEAD via src/star/pgo/pgo_build.sh, trained on pbmc2m; installs bin/singlet-accel-TAG.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
TAG=$1; FLAGS=$2; export CMAKE_ARGS=${3:-}
BR=perf/star-accel
git -C $W/repos/singlet-py.git fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
SRC=$W/src/singlet-py/perf__star-accel
git -C $SRC reset -q --hard refs/heads/$BR; git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
REF=/mnt/projects/debruinz_project/cellarium/reference
cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train.1fq
export TMPDIR=$SCR
bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train.1fq $REF/GRCh38-2024-A/star_2.7.11b \
  $REF/GRCh38-2024-A/genes/genes.gtf $REF "$FLAGS" || { for f in $SCR/b.*.log; do echo == $f; tail -n 15 $f; done; exit 1; }
grep -h "STAR_PROFILE\|SEQCACHE: look" $SCR/b.train.log || true
cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-accel-$TAG; echo "installed singlet-accel-$TAG"
