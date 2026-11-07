#!/bin/bash
#SBATCH --job-name=if-starbuild
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# STAR 2.7.11b with a per-UMI Velocyto dump (STAR_VELO_DUMP) -> pipeline-benchmark/bin/STAR-velodump
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
D=$W/src/STAR-2.7.11b-velodump
rm -rf $D; git clone -q --depth 1 --branch 2.7.11b https://github.com/alexdobin/STAR.git $D
$ENV/bin/python $W/jobs/intronfix/star_velodump_patch.py $D/source/SoloFeature_countVelocyto.cpp
export PATH=$ENV/bin:$PATH CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
cd $D/source
make -j 16 STAR CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXXFLAGS_SIMD=-mavx2 > $D/build.log 2>&1 || { tail -40 $D/build.log; exit 1; }
cp STAR $PB/bin/STAR-velodump
LD_LIBRARY_PATH=$ENV/lib $PB/bin/STAR-velodump --version
sha256sum $PB/bin/STAR-velodump
