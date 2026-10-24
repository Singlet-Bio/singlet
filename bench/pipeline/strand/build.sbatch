#!/bin/bash
#SBATCH --job-name=pb-strand-build
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build singlet from fix/strand-detection ($BIN_STRAND), then build and run the unit
# tests it touches: strand detection, molecule labels, protocol detection.
#   sbatch build.sbatch [all|tests]
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__strand-detection/bench/pipeline/strand/common.sh
export CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}
export CMAKE_POLICY_VERSION_MINIMUM=3.5
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
WHAT=${1:-all}
if [ "$WHAT" = all ]; then
  S=$SCR/src; B=$SCR/build
  rsync -a --exclude .git --exclude build "$SRC_STRAND/" $S/
  grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
  cmake -S $S -B $B -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release > $B.cmake.log 2>&1 || { tail -40 $B.cmake.log; exit 1; }
  cmake --build $B --target singlet -j $SLURM_CPUS_PER_TASK > $B.build.log 2>&1 || { grep -m20 -B2 -A8 "error" $B.build.log; exit 1; }
  cp $B/src/pipeline/singlet $BIN_STRAND
  echo "[build] $BIN_STRAND: $($BIN_STRAND --version 2>&1 | head -1)"
fi
T=$SCR/build-tests
TESTS="test_strand_detect test_molecule_labels test_protocol_confidence_override test_ws2_protocol_hardening test_indrop_detect test_seqwell_detect test_species_short_r2 test_1fq_header"
cmake -S $SRC_STRAND -B $T -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $T.cmake.log 2>&1 || { tail -40 $T.cmake.log; exit 1; }
for t in $TESTS; do
  cmake --build $T --target $t -j 8 > $T.$t.log 2>&1 || { echo "[tests] BUILD FAIL $t"; grep -m30 -A6 "error" $T.$t.log; continue; }
done
$T/tests/cpp/test_strand_detect | tail -70
cd $T && ctest -R "$(echo $TESTS | tr ' ' '|')" --output-on-failure 2>&1 | tail -60
