#!/bin/bash
#SBATCH --job-name=alpha-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Research Alpha benchmark B: build the singlet binary at the commit every published Alpha sample reports in its
# summary.json (git_sha 8e1803ba = v0.3.0-pilot-freeze, the campaign binary), with the build patches the pipeline
# benchmark used for the atlas-engine baseline (x86-64-v3 instead of -march=native; htslib symbol shim).
# Output: $WORK/alpha_bench/bin/singlet-alpha-8e1803b
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/pbench
SHA=8e1803ba5ac9cb8038e666f06df446c9a85550b0
OUTB=$WORK/alpha_bench/bin; mkdir -p $OUTB
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-} CMAKE_POLICY_VERSION_MINIMUM=3.5
git clone -q --filter=blob:none --no-checkout git@github.com:Singlet-Bio/singlet.git $SCR/src
git -C $SCR/src checkout -q $SHA
echo "[build] source $(git -C $SCR/src rev-parse HEAD) $(git -C $SCR/src describe --tags --always 2>/dev/null)"
echo "[build] pileup/count source differences vs the pipeline benchmark's atlas-engine baseline b1335a8:"
git -C $SCR/src diff --stat $SHA b1335a8 -- src/pipeline src/pileup 2>/dev/null | tail -5 || true
grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
[ -f $SCR/src/src/star/bam_cat.c ] && sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release > $SCR/cmake.log 2>&1 || { tail -40 $SCR/cmake.log; exit 1; }
cmake --build $SCR/b --target singlet -j $SLURM_CPUS_PER_TASK > $SCR/build.log 2>&1 || { grep -m20 -B2 -A8 "error" $SCR/build.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $OUTB/singlet-alpha-8e1803b
ln -sfn $WORK/pipeline-benchmark/whitelists $WORK/alpha_bench/whitelists
$OUTB/singlet-alpha-8e1803b --version 2>&1 | head -2
$OUTB/singlet-alpha-8e1803b --help 2>&1 | grep -E -- "--(max-reads|output-format|raw-matrix|cascade|snps|te-classify|pipeline)" | head -20
sha256sum $OUTB/singlet-alpha-8e1803b
