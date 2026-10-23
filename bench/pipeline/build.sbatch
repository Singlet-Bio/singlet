#!/bin/bash
#SBATCH --job-name=pb-build
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build the singlet binary from the baseline (b1335a8 = atlas counting code) and the
# fixed branch, then build and run the C++ unit tests of the fixed branch.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
export CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
WHAT=${1:-all}   # all | tests
build_bin() {  # $1 src  $2 out-binary
  local b=$SCR/build-$(basename $2)
  # x86-64-v3 instead of -march=native so the binary runs on every Clipper CPU node
  rsync -a --exclude .git --exclude build "$1/" $SCR/src-$(basename $2)/
  sed -i 's/-march=native/-march=x86-64-v3/' $SCR/src-$(basename $2)/src/pipeline/CMakeLists.txt
  cmake -S $SCR/src-$(basename $2) -B $b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release > $b.cmake.log 2>&1 || { tail -40 $b.cmake.log; return 1; }
  cmake --build $b --target singlet -j $SLURM_CPUS_PER_TASK > $b.build.log 2>&1 || { grep -m20 -B2 -A8 "error" $b.build.log; return 1; }
  cp $b/src/pipeline/singlet $2
  echo "[build] $2: $($2 --version 2>&1 | head -1)"
}
mkdir -p $PB/bin
ln -sfn $SRC_NEW/python/singlet/data/whitelists $PB/whitelists   # <bin>/../whitelists
if [ "$WHAT" = all ]; then
  build_bin $SRC_OLD $BIN_OLD &
  build_bin $SRC_NEW $BIN_NEW &
  wait
fi
# Unit tests (fixed branch)
T=$SCR/build-tests
cmake -S $SRC_NEW -B $T -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $T.cmake.log 2>&1 || { tail -40 $T.cmake.log; exit 1; }
for t in test_molecule_labels test_saturation test_cell_qc test_em_rescue test_sparse_accumulator test_read_stats; do
  cmake --build $T --target $t -j 8 > $T.$t.log 2>&1 || { echo "[tests] BUILD FAIL $t"; grep -m30 -A6 "error" $T.$t.log; continue; }
done
cd $T && ctest -R "test_molecule_labels|test_saturation|test_cell_qc|test_em_rescue|test_sparse_accumulator|test_read_stats" --output-on-failure 2>&1 | tail -80
