#!/bin/bash
#SBATCH --job-name=cs-build
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only singlet (-DSINGLET_ENGINE_GPU=OFF, -march=x86-64-v3, -g) of a branch tip of the Singlet-Bio/singlet bare clone, plus the samprof
# sampler. -g adds symbols only (GCC emits the same code with and without it), so one binary serves the timing, the identity gate and the profile.
# Env: BR (branch), optional NAME (output dir, default the commit's first 8 characters).
# Output: $W/jobs/cpu-speed/bin/$NAME/{singlet-cpu, singlet-gpu -> singlet-cpu (the name the campaign gate calls), libsamprof.so, commit}.
# The whitelists sit at bin/../whitelists, where singlet finds them.
set -euo pipefail
: "${BR:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$BR | cut -f1); [ -n "$WANT" ] || { echo "no branch $BR"; exit 3; }
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/cs/build-$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
COMMIT=$(git -C $BARE rev-parse refs/cs/build-$SLURM_JOB_ID); [ "$COMMIT" = "$WANT" ] || { echo "fetched $COMMIT, tip is $WANT"; exit 3; }
NAME=${NAME:-${COMMIT:0:8}}; OUTD=$W/jobs/cpu-speed/bin/$NAME
[ -e $OUTD/singlet-cpu ] && { echo "exists: $OUTD"; exit 0; }; mkdir -p $OUTD
SCR=/tmp/$USER-csb-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/cs/build-$SLURM_JOB_ID | tar -x -C $SCR/src; S=$SCR/src
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=OFF -DCMAKE_CXX_FLAGS=-g -DCMAKE_C_FLAGS=-g > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target singlet -j ${SLURM_CPUS_PER_TASK:-24} > $SCR/b.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b.log || tail -40 $SCR/b.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $OUTD/singlet-cpu
\$CC -O2 -fPIC -shared -o $OUTD/libsamprof.so $S/bench/cpu-speed/samprof/samprof.c -ldl -lrt
EOS
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh
ln -sfn singlet-cpu $OUTD/singlet-gpu; echo $COMMIT > $OUTD/commit; echo $COMMIT > $OUTD/commit-gpu
ln -sfn $W/pipeline-benchmark/whitelists $W/jobs/cpu-speed/bin/whitelists 2>/dev/null || true
sha256sum $OUTD/singlet-cpu; echo "[build] $NAME ($COMMIT) ok in $(( $(date +%s) - T0 )) s"
