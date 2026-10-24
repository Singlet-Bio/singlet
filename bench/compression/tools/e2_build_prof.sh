#!/usr/bin/env bash
# build the sampling-profiler variant of e2_bench (with line tables)
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -lwinmm -static"
g++ -std=c++20 -O3 -march=native -mtune=native -DNDEBUG -g1 -DE2_PROF_BUILD -Wall -Wno-unused-function -I$ROOT/include -I$TC/include -Iharness -Ikernels harness/e2_bench.cpp -o bin/e2_bench_prof.exe $LIBS 2>&1 | grep -E "error" 
echo built
