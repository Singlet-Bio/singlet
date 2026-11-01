#!/bin/bash
#SBATCH --job-name=cpuperf-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only singlet (no CUDA) of COMMIT (default d9834b8a), plus a -DSTAR_PROFILE variant, for the CPU-engine profile (docs/research-beta/cpu-engine-perf.md).
# Same toolchain as scripts/engine_gpu/build_clipper.sbatch. Output: $W/jobs/cpu-perf/<COMMIT8>/{bin/singlet-cpu,bin/singlet-cpu-prof,whitelists -> pipeline-benchmark/whitelists}.
# The whitelists sit next to the binary (bin/../whitelists), which is what singlet auto-discovers (no SINGLET_WHITELIST_DIR).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git
[ -n "${BRANCH:-}" ] && { git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/cpuperf/build; COMMIT=refs/cpuperf/build; }   # BRANCH=claude/cpu-perf-c66d3f builds its tip
COMMIT=$(git -C $BARE rev-parse ${COMMIT:-d9834b8a}); OUTD=$W/jobs/cpu-perf/${COMMIT:0:8}; mkdir -p $OUTD/bin
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src; S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
for v in cpu cpu-prof; do
  X=""; [ \$v = cpu-prof ] && X=-DSTAR_PROFILE
  cmake -S $S -B $SCR/b-\$v -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=OFF -DCMAKE_CXX_FLAGS="\$X" > $SCR/c-\$v.log 2>&1 || { tail -40 $SCR/c-\$v.log; exit 1; }
  cmake --build $SCR/b-\$v --target singlet -j 24 > $SCR/b-\$v.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b-\$v.log || tail -40 $SCR/b-\$v.log; exit 1; }
  cp $SCR/b-\$v/src/pipeline/singlet $OUTD/bin/singlet-\$v.new && mv $OUTD/bin/singlet-\$v.new $OUTD/bin/singlet-\$v
done
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh
ln -sfn $W/pipeline-benchmark/whitelists $OUTD/whitelists; echo $COMMIT > $OUTD/commit
sha256sum $OUTD/bin/*; ls $OUTD/whitelists | head -3; echo "[build] ok"
