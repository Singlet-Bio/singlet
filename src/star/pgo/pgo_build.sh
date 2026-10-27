#!/bin/bash
# Profile-guided build of the bundled STAR (the rest of singlet is built normally).
#
#   src/star/pgo/pgo_build.sh SRC_DIR BUILD_DIR TRAIN.1fq[:TRAIN2.1fq...] GENOME_DIR GTF REF_BASE [extra STAR flags]
#
# 1. builds singlet with the STAR objects instrumented (-fprofile-generate),
# 2. runs the pipeline once on each training .1fq (a ~2M-read 10x sample each is enough: the profile is
#    about which branches the aligner takes, not about volume; give a 3' and a 5' sample, whose
#    too-short and soft-clip paths differ; gcc sums the counts of all runs),
# 3. rebuilds the STAR objects with -fprofile-use in the same BUILD_DIR (gcc finds the .gcda files
#    next to the objects, so both builds must use the same directory).
# The result is BUILD_DIR/src/pipeline/singlet. Profiles are build products: not committed.
# Env: CMAKE_ARGS (extra cmake arguments), JOBS (parallel build jobs, default 16), THREADS (4).
set -euo pipefail
SRC=$1; B=$2; TRAIN=$3; GENOME=$4; GTF=$5; REFBASE=$6; FLAGS=${7:-}
JOBS=${JOBS:-16}; THREADS=${THREADS:-4}

cmake -S "$SRC" -B "$B" -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-} \
  "-DSINGLET_STAR_EXTRA_FLAGS=$FLAGS -fprofile-generate -fprofile-update=atomic -DSINGLET_PGO_GENERATE" \
  "-DCMAKE_EXE_LINKER_FLAGS=-fprofile-generate" > "$B.cmake-gen.log" 2>&1
find "$B" -name '*.gcda' -delete
cmake --build "$B" --target singlet -j "$JOBS" > "$B.build-gen.log" 2>&1

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
"$B/src/pipeline/singlet" "$TRAIN" --ref-base "$REFBASE" --genome-dir "$GENOME" --exons "$GTF" \
  --out-prefix "$T/" --threads "$THREADS" --pipeline --output-format mtx > "$B.train.log" 2>&1
n=$(find "$B" -name '*.gcda' | wc -l)
echo "pgo: training run wrote $n profile files"
[ "$n" -gt 0 ] || { echo "pgo: no profile written"; exit 1; }

cmake -S "$SRC" -B "$B" "-DSINGLET_STAR_EXTRA_FLAGS=$FLAGS -fprofile-use -fprofile-partial-training -Wno-missing-profile -Wno-error=coverage-mismatch" \
  "-DCMAKE_EXE_LINKER_FLAGS=" > "$B.cmake-use.log" 2>&1
cmake --build "$B" --target singlet -j "$JOBS" > "$B.build-use.log" 2>&1
echo "pgo: built $B/src/pipeline/singlet"
