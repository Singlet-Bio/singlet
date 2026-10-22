#!/usr/bin/env bash
# -O3 build of the pp bench to an arbitrary exe name (so a running benchmark does not lock the output):
#   bash tools/idea4_pp_build_o3.sh bin/idea4_pp_b2.exe
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
g++ -std=c++20 -O3 -march=native -mtune=native -DNDEBUG -w -I$ROOT/include -I$TC/include -Iharness -Ikernels harness/idea4_pp_bench.cpp -o "$1" $LIBS && echo "o3 build done $1"
