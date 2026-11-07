#!/bin/bash
#SBATCH --job-name=dbl-ctest
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Host ctest of a branch: CPU build with tests (no CUDA), whole node. BRANCH below.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BRANCH=fix/doublet-expression-score
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/dblct/head
COMMIT=$(git -C $BARE rev-parse refs/dblct/head); echo "commit $COMMIT"
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src; S=$SCR/src; NC=$(nproc)
grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
cat > $SCR/t.sh <<EOS
set -uo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export CPLUS_INCLUDE_PATH=$ENV/include C_INCLUDE_PATH=$ENV/include LIBRARY_PATH=$ENV/lib
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_BUILD_TESTS=ON -DSINGLET_ENGINE_GPU=OFF > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
cmake --build $SCR/b -j $NC -- -k > $SCR/b.log 2>&1 || { grep -m20 -B2 -A6 "error" $SCR/b.log | head -60; }
cd $SCR/b && ctest -j 16 --timeout 900 --output-on-failure 2>&1 | tail -40
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/t.sh
