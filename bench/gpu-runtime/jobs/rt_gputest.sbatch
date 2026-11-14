#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:quadro_rtx_8000:1 -c 16 --mem=64G --time=01:30:00 -J rt-gputest -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-gputest: the GPU unit tests TESTS="..." of BASE + PATCH (as rt_build.sbatch) built in the container (pipeline + tests), run on this node's GPU. NAME=<label>
set -uo pipefail
: "${BASE:?}" "${PATCH:?}" "${NAME:?}"; TESTS=${TESTS:-test_gpu_pileup_stream}
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; OUTD=$W/jobs/gpu-runtime/gputest/$NAME; mkdir -p $OUTD
COMMIT=$(git -C $BARE rev-parse --verify "$BASE^{commit}")
SCR=/tmp/$USER-rtgt-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src; S=$SCR/src
( cd $S && git apply --stat --apply -p1 "$PATCH" ) || { echo "patch does not apply"; exit 3; }
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -uo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=ON -DCMAKE_CXX_FLAGS=-I$ENV/include \
  -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="75" > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target $TESTS -j 16 > $OUTD/build.log 2>&1 || { grep -m40 -B3 -A10 "error" $OUTD/build.log || tail -60 $OUTD/build.log; exit 1; }
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh || { echo "GPU-TEST BUILD FAILED" | tee $OUTD/result.txt; exit 1; }
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib SINGLET_ENGINE_GPU_MIN_CC=70
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee $OUTD/run.txt
for t in $TESTS; do
  b=$(find $SCR/b -type f -name $t -perm -u+x | head -1); [ -n "$b" ] || { echo "no binary $t"; continue; }
  ( cd $SCR && $b > $OUTD/$t.txt 2>&1 ); echo "TEST $t exit $?" | tee -a $OUTD/run.txt; cut -c1-300 $OUTD/$t.txt | tail -60
done
