#!/bin/bash
#SBATCH --job-name=cpuperf-ctest
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Configure with SINGLET_BUILD_TESTS=ON (CPU only), build everything, run ctest. BRANCH (default claude/cpu-perf-c66d3f) or COMMIT.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BR=${BRANCH:-claude/cpu-perf-c66d3f}; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/cpuperf/ctest
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/cpuperf/ctest | tar -x -C $SCR/src; S=$SCR/src; echo "commit $(git -C $BARE rev-parse refs/cpuperf/ctest)"
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
cat > $SCR/run.sh <<EOS
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON -DSINGLET_ENGINE_GPU=OFF > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
cmake --build $SCR/b -j 24 -k > $SCR/b.log 2>&1; echo "build exit \$?"; grep -m20 "error" $SCR/b.log
cd $SCR/b && ctest -j 12 --timeout 600 --output-on-failure 2>&1 | tail -40
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/run.sh
