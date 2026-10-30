#!/bin/bash
#SBATCH --job-name=ar-native
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build the fixed engine with -march=native on a short-cpu node (Cascade Lake), for perf_run t4_native.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
SRC=$W/src/singlet-py/fix__molecule-labels
export PATH=$ENV/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib
export CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release > $SCR/cmake.log 2>&1 || { tail -30 $SCR/cmake.log; exit 1; }
cmake --build $SCR/b --target singlet -j 16 > $SCR/build.log 2>&1 || { grep -m20 -A5 error $SCR/build.log; exit 1; }
cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-fixed-native; ls -la $PB/bin/
