#!/usr/bin/env bash
# Build cpp/decode_bench and cpp/kernel_bench on Linux with the system (or conda) g++ and zstd. decode_bench has no -march
# flag (SIMD comes from pz2's runtime dispatch); kernel_bench is built for x86-64-v3 (AVX2) so the textbook codes it
# compares against get the same vector width as the pz2 AVX2 kernel.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../../../.." && pwd)"   # repository root (include/singlet/pz2)
CXX="${CXX:-g++}"
INC=(-I"$ROOT/include" ${CONDA_PREFIX:+-I"$CONDA_PREFIX/include" -L"$CONDA_PREFIX/lib"})
"$CXX" -std=c++17 -O2 -DNDEBUG "${INC[@]}" "$HERE/decode_bench.cpp" -o "$HERE/decode_bench" -lzstd -lpthread
echo "built $HERE/decode_bench"
"$CXX" -std=c++17 -O3 -march=x86-64-v3 -DNDEBUG "${INC[@]}" "$HERE/kernel_bench.cpp" -o "$HERE/kernel_bench" -lzstd -lpthread
echo "built $HERE/kernel_bench"
