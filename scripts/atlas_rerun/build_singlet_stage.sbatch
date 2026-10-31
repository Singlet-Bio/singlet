#!/bin/bash
#SBATCH --job-name=s9-build
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build the singlet binary with and without the GPU engine on Clipper (docs/engine-gpu.md has the generic recipe).
# Toolchain: the pbench conda env (compilers, htslib, ncbi-vdb, zstd, hdf5) inside the CUDA 12.4 devel image (nvcc).
# S8: prod/campaign tip; also builds singlet-prof (-DSTAR_PROFILE). BRANCH (default gpu/engine-gpu) is fetched by URL into a private ref. Outputs: $W/jobs/w11/engine/bin/singlet-{gpu,cpu}.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
TAG=${TAG:-campaign}; BARE=$W/repos/singlet-py.git; BR=${BRANCH:-claude/stage-1fg2}; OUTD=${OUTD:-$W/jobs/s9/bin}; mkdir -p $OUTD
VARIANTS=${VARIANTS:-cpu}; XFLAGS=; OUTD=${OUTD:-$W/jobs/s9/bin}; mkdir -p $OUTD
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/s9/$TAG
COMMIT=$(git -C $BARE rev-parse refs/s9/$TAG); echo "COMMIT=$COMMIT"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/s9/$TAG | tar -x -C $SCR/src
S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
# the conda libstdc++ (gcc 16) before the CUDA image's system one, which CUDA's implicit link directories would otherwise put first
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
\$CXX --version | head -1; nvcc --version | tail -1
for v in $VARIANTS; do
  E=OFF; XF="$XFLAGS"; [ \$v = prof ] && XF="-DSTAR_PROFILE"
  cmake -S $S -B $SCR/b-\$v -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=\$E -DCMAKE_CXX_FLAGS="\$XF" \
    -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="80;90" > $SCR/c-\$v.log 2>&1 \
    || { tail -40 $SCR/c-\$v.log; exit 1; }
  grep -i "GPU engine" $SCR/c-\$v.log || true
  cmake --build $SCR/b-\$v --target singlet singlet-stage -j 8 > $SCR/b-\$v.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b-\$v.log || tail -40 $SCR/b-\$v.log; exit 1; }
  cp $SCR/b-\$v/src/pipeline/singlet $OUTD/singlet-\$v.new && mv $OUTD/singlet-\$v.new $OUTD/singlet-\$v
  cp $SCR/b-\$v/singlet-stage $OUTD/singlet-stage-\$v
done
EOS
apptainer exec --nv -B $W -B $SCR $IMG bash $SCR/build.sh
echo "$COMMIT" > $OUTD/commit; sha256sum $OUTD/singlet-*
echo "[build] ok"
