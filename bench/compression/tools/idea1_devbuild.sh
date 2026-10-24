#!/usr/bin/env bash
# Dev build for the Idea 1 bench with a precompiled common.h (same flags as build.sh) so rebuilds are fast.
#   bash tools/idea1_devbuild.sh [harness/idea1_bench.cpp] [out.exe]
set -uo pipefail
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
SRC=${1:-harness/idea1_bench.cpp}
OUT=${2:-bin/idea1_bench.exe}
CXXFLAGS="-std=c++20 -O3 -march=native -mtune=native -DNDEBUG -Wall -Wno-unused-function -Wno-deprecated-declarations -I$ROOT/include -I$TC/include -Iharness -Ikernels"
CXXFLAGS_P="-Iresults/idea1_pch $CXXFLAGS"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
mkdir -p results/idea1_pch
if [ ! -f results/idea1_pch/common.h.gch ] || [ harness/common.h -nt results/idea1_pch/common.h.gch ]; then
  cp harness/common.h results/idea1_pch/common.h
  echo "[devbuild] precompiling common.h"
  g++ $CXXFLAGS -x c++-header results/idea1_pch/common.h -o results/idea1_pch/common.h.gch || exit 1
fi
echo "[devbuild] $SRC -> $OUT"
cp "$SRC" results/idea1_pch/dev_main.cpp
g++ $CXXFLAGS_P results/idea1_pch/dev_main.cpp -o "$OUT" $LIBS 2>&1 | grep -E "error|Error|warning: unused" ; echo "[devbuild] done rc=${PIPESTATUS[0]}"
