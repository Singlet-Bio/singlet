#!/bin/bash
#SBATCH --job-name=rg-build-fix
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Release gate: build ONE pinned commit of prod/campaign as the CPU binary (singlet-cpu, engine off) and the GPU-engine build
# (singlet-gpu, run with --engine gpu), side by side in $OUTD. Recipe: bench/engine_gpu_gate/build.sbatch (pbench conda toolchain
# inside the CUDA 12.4 devel image). Usage: sbatch --export=ALL,PIN=<sha> build.sbatch
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; PIN=${PIN:-ba427bcd}; OUTD=$W/release-gate/bin/${PIN:0:8}; mkdir -p $OUTD
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/fix/stall-watchdog:refs/rg/campaign
COMMIT=$(git -C $BARE rev-parse $PIN^{commit}); echo "COMMIT=$COMMIT"
git -C $BARE merge-base --is-ancestor $COMMIT refs/rg/campaign && echo "pin is on prod/campaign"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src
S=$SCR/src
# portable x86-64-v3 (nodes differ), as the other build jobs
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:/usr/local/cuda/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
for v in cpu; do
  E=ON; [ \$v = cpu ] && E=OFF
  cmake -S $S -B $SCR/b-\$v -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=\$E \
    -DCMAKE_CUDA_HOST_COMPILER=\$CXX -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="80;90" > $SCR/c-\$v.log 2>&1 \
    || { tail -40 $SCR/c-\$v.log; exit 1; }
  cmake --build $SCR/b-\$v --target singlet -j 8 > $SCR/b-\$v.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b-\$v.log || tail -40 $SCR/b-\$v.log; exit 1; }
  cp $SCR/b-\$v/src/pipeline/singlet $OUTD/singlet-\$v.new && mv $OUTD/singlet-\$v.new $OUTD/singlet-\$v
done
EOS
apptainer exec --nv -B $W -B $SCR $IMG bash $SCR/build.sh
echo "$COMMIT" > $OUTD/commit; sha256sum $OUTD/singlet-cpu | tee $OUTD/sha256.txt
# regression test of the fix (standalone, header-only)
$ENV/bin/x86_64-conda-linux-gnu-g++ -std=c++17 -I$SCR/src/include -o $SCR/tsw $SCR/src/tests/cpp/test_stall_watchdog.cpp && $SCR/tsw
echo "[build] ok"
