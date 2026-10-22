#!/usr/bin/env bash
# Build one idea3 harness to a differently named binary (so a running bin/idea3_*.exe is not touched).
# usage: bash tools/idea3_build_dev.sh harness/idea3_bench.cpp bin/idea3_bench_dev.exe
set -uo pipefail
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
CXXFLAGS="-std=c++20 -O3 -march=native -mtune=native -DNDEBUG -Wall -Wno-unused-function -I$ROOT/include -I$TC/include -Iharness -Ikernels"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
g++ $CXXFLAGS "$1" -o "$2" $LIBS && echo "[build] ok $2"
