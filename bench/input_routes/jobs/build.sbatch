#!/bin/bash
#SBATCH --job-name=ir-build
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# input-routes: build singlet (--engine gpu) from a branch of Singlet-Bio/singlet (copy of bench/gpu_next/jobs/build.sbatch). Needs BR and NAME.
#   sbatch --export=ALL,BR=feat/input-routes,NAME=<sha8> bench/input_routes/jobs/build.sbatch
# Output: $W/jobs/input-routes/bin/$NAME/singlet-gpu and commit. The binary is named by its directory and records its commit. Nothing existing is overwritten.
# Recipe of bench/fable/release/jobs/build.sbatch (pbench toolchain in the CUDA 12.4 devel image). EXTRA_CXX adds C++ flags (e.g. -DSTAR_PROFILE).
set -euo pipefail
: "${BR:?}" "${NAME:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; OUTD=$W/jobs/input-routes/bin/$NAME; [ -e $OUTD/singlet-gpu ] && { echo "refusing to overwrite $OUTD"; exit 2; }; mkdir -p $OUTD
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$BR | cut -f1); [ -n "$WANT" ] || { echo "no branch $BR"; exit 3; }
# one fetch at a time (concurrent builds share the bare clone), into a ref named by NAME, then check it is the branch tip that was asked for
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/ir/$NAME ) 9>$BARE/.gn-fetch.lock
COMMIT=$(git -C $BARE rev-parse refs/ir/$NAME); echo "COMMIT=$COMMIT NAME=$NAME"
[ "$COMMIT" = "$WANT" ] || { echo "fetched $COMMIT, branch tip is $WANT"; exit 3; }
SCR=/tmp/$USER-gnb-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/ir/$NAME | tar -x -C $SCR/src
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
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=ON -DCMAKE_CXX_FLAGS="${EXTRA_CXX:-}" \
  -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="90" > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target singlet -j ${SLURM_CPUS_PER_TASK:-24} > $SCR/b.log 2>&1 || { grep -m40 -B3 -A10 "error" $SCR/b.log || tail -60 $SCR/b.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $OUTD/singlet-gpu
EOS
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh
echo "$COMMIT" > $OUTD/commit-gpu; ln -sfn $W/pipeline-benchmark/whitelists $OUTD/../whitelists 2>/dev/null || true
sha256sum $OUTD/singlet-gpu; ldd $OUTD/singlet-gpu | grep -iE "cuda|not found" || true
echo "[build] $NAME ok in $(( $(date +%s) - T0 )) s"
