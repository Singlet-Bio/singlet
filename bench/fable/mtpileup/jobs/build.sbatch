#!/bin/bash
#SBATCH --job-name=e23-build
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E23 (docs/gpu-engine/fable/E23-mtpileup.md; copy of E19's bench/fable/release/jobs/build.sbatch): the singlet binary (--engine gpu) from gpu/fable/mtpileup (BRANCH),
# at REV (a commit of that branch; default its head), recipe of E8's build_singlet.sbatch. VARIANT=base: the same build as gpu, named singlet-base (E23's
# baseline: the branch's instrumentation commit, before the fix)
# (pbench conda toolchain in the CUDA 12.4 devel image). VARIANT=gpu (default) -> bin/singlet-gpu; VARIANT=prof -> bin/singlet-prof built with
# -DSTAR_PROFILE (STAR's per-phase cycle counters in Log.out). Built on short-gpu without a GPU request (short-cpu is usually queued).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/mtpileup}; V=${VARIANT:-gpu}; OUTD=${OUTD:-$W/jobs/fable/mtpileup/bin}; mkdir -p $OUTD
XF=""; [ $V = prof ] && XF="-DSTAR_PROFILE"; REV=${REV:-refs/fable/mtpileup-$V}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/mtpileup-$V
COMMIT=$(git -C $BARE rev-parse "$REV^{commit}"); echo "COMMIT=$COMMIT VARIANT=$V"
SCR=/tmp/$USER-e23b-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src
S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
\$CXX --version | head -1; nvcc --version | tail -1
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=ON -DCMAKE_CXX_FLAGS="$XF" \
  -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="80;90" > $SCR/c.log 2>&1 \
  || { tail -40 $SCR/c.log; exit 1; }
grep -i "GPU engine" $SCR/c.log || true
cmake --build $SCR/b --target singlet -j ${SLURM_CPUS_PER_TASK:-24} > $SCR/b.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b.log || tail -40 $SCR/b.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $OUTD/singlet-$V.new && mv $OUTD/singlet-$V.new $OUTD/singlet-$V
EOS
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh
echo "$COMMIT" > $OUTD/commit-$V; sha256sum $OUTD/singlet-$V; ldd $OUTD/singlet-$V | grep -iE "cuda|not found" || true
echo "[build] $V ok in $(( $(date +%s) - T0 )) s"
