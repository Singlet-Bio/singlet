#!/bin/bash
#SBATCH --job-name=t1k-patch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-101 HLA: a T1K install with singlet's speed patches (bench/hla/patches/, same outputs as T1K as released) next to the
# released one. FROM's index and scripts are copied as they are (same IPD-IMGT/HLA release, no rebuild); only the three binaries
# are rebuilt from FROM's T1K commit with the patches applied, with the toolchain and flags of t1k-setup.sbatch. The manifest gains
# "patches" (file, sha256), "patched" and "patch_builder".
#   BRANCH: singlet branch that holds the patches (fetched by URL into a private ref of the bare repo)
#   PATCHES: space-separated patch files under bench/hla/patches/ (default: all, in name order)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench
BARE=$W/repos/singlet-py.git; BR=${BRANCH:?singlet branch with bench/hla/patches}
FROM=${FROM:-$REF/hla/t1k}; DEST=${DEST:?destination, e.g. $REF/hla/t1k-p1}
SCR=/tmp/$USER-t1kp-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
[ -e $DEST ] && { echo "refusing to overwrite $DEST"; exit 2; }
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/t1kp/$SLURM_JOB_ID
mkdir -p $SCR/singlet; git -C $BARE archive refs/t1kp/$SLURM_JOB_ID bench/hla/patches | tar -x -C $SCR/singlet
SCOMMIT=$(git -C $BARE rev-parse refs/t1kp/$SLURM_JOB_ID); git -C $BARE update-ref -d refs/t1kp/$SLURM_JOB_ID
P=$SCR/singlet/bench/hla/patches
[ -n "${PATCHES:-}" ] || PATCHES=$(cd $P && ls *.patch | sort)
TAG=$(grep -o '"tag": "[^"]*"' $FROM/manifest.json | cut -d'"' -f4); COMMIT=$(grep -o '"commit": "[^"]*"' $FROM/manifest.json | cut -d'"' -f4)
git clone -q --depth 1 --branch $TAG https://github.com/mourisl/T1K.git $SCR/T1K
[ "$(git -C $SCR/T1K rev-parse HEAD)" = "$COMMIT" ] || { echo "T1K $TAG is not commit $COMMIT"; exit 3; }
PJSON=""
for p in $PATCHES; do
  git -C $SCR/T1K apply $P/$p
  PJSON="$PJSON${PJSON:+, }{\"file\": \"bench/hla/patches/$p\", \"sha256\": \"$(sha256sum $P/$p | cut -c1-64)\"}"
done
git -C $SCR/T1K diff --stat
cd $SCR/T1K
make -j8 CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LINKFLAGS="-lpthread -lz -L$ENV/lib -Wl,-rpath,$ENV/lib" CXXFLAGS="-O3 -g -I$ENV/include" fastq-extractor genotyper analyzer
cp -r $FROM $DEST.tmp-$SLURM_JOB_ID
cp fastq-extractor genotyper analyzer $DEST.tmp-$SLURM_JOB_ID/bin/
M=$DEST.tmp-$SLURM_JOB_ID/manifest.json
sed -i "\$ s|}\$|,\n \"patches\": [$PJSON], \"patched\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"patch_builder\": \"bench/hla/jobs/t1k-patch-install.sbatch at singlet $SCOMMIT\"}|" $M
cat $M
mv $DEST.tmp-$SLURM_JOB_ID $DEST
sha256sum $FROM/bin/* $DEST/bin/* | sort
ls -l $DEST $DEST/bin
