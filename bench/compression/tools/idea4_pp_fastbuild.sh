#!/usr/bin/env bash
# quick -O1 build of the pp bench for debugging: bash tools/idea4_pp_fastbuild.sh -> bin/idea4_pp_dbg.exe
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
g++ -std=c++20 -O1 -march=native -DNDEBUG -w -I$ROOT/include -I$TC/include -Iharness -Ikernels harness/idea4_pp_bench.cpp -o bin/idea4_pp_dbg.exe $LIBS
echo "dbg build done" 
