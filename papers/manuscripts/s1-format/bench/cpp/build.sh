#!/usr/bin/env bash
# Build cpp/decode_bench on Linux with the system (or conda) g++ and zstd. No -march flag: SIMD comes from runtime dispatch.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"   # repository root (include/singlet/pz2)
CXX="${CXX:-g++}"
"$CXX" -std=c++17 -O2 -DNDEBUG -I"$ROOT/include" ${CONDA_PREFIX:+-I"$CONDA_PREFIX/include" -L"$CONDA_PREFIX/lib"} \
    "$HERE/decode_bench.cpp" -o "$HERE/decode_bench" -lzstd -lpthread
echo "built $HERE/decode_bench"
