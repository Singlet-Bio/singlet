#!/bin/bash
#SBATCH --job-name=w11-build-timers
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 host-time study: PGO build of prod/campaign (28969b87) + timers.patch -> $W/jobs/w11/host/bin/singlet-timers.
# Same recipe as jobs/campaign/build_campaign.sbatch, except: the source is `git archive` of the pinned commit
# (the shared prod__campaign worktree is never reset or touched) and the output is a DEV binary in our own folder.
# Never writes $W/pipeline-benchmark/bin/*.
set -euo pipefail
trap 'echo "[build] failed at line $LINENO: $BASH_COMMAND"' ERR
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
BARE=$W/repos/singlet-py.git; COMMIT=28969b87b2ca237f9a2344792e04b59f3cabf3e9
H=$W/jobs/w11/host; mkdir -p $H/bin
git -C $BARE cat-file -e $COMMIT^{commit} 2>/dev/null || git -C $BARE fetch -q origin +refs/heads/prod/campaign:refs/w11/campaign
git -C $BARE cat-file -e $COMMIT^{commit}
echo "COMMIT=$COMMIT"
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src
cd $SCR/src && patch -p1 --no-backup-if-mismatch < $H/timers.patch
grep -c W11S src/pipeline/singlet.cpp include/singlet/pileup/export.h include/singlet/pileup/donor_demux.h
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $SCR/src/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
REF=/mnt/projects/debruinz_project/cellarium/reference
cp $PB/data/pbmc2m/pbmc2m.1fq $SCR/train.1fq
export TMPDIR=$SCR CMAKE_ARGS="-DSINGLET_GIT_COMMIT=$COMMIT"
bash $SCR/src/src/star/pgo/pgo_build.sh $SCR/src $SCR/b $SCR/train.1fq $REF/GRCh38-2024-A/star_2.7.11b \
  $REF/GRCh38-2024-A/genes/genes.gtf $REF "" || { for f in $SCR/b.*.log; do echo == $f; tail -n 25 $f; done; exit 1; }
cp $SCR/b/src/pipeline/singlet $H/bin/singlet-timers.new && mv $H/bin/singlet-timers.new $H/bin/singlet-timers
sha256sum $H/bin/singlet-timers
echo "[build] done"
