#!/usr/bin/env bash
# usage: tools/track_b_build.sh src.cpp out.exe [extra g++ flags...]   (same flags as build.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
src=$1; out=$2; shift 2
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
g++ -std=c++20 -O3 -march=native -mtune=native -DNDEBUG -Wall -Wno-unused-function -Wno-deprecated-declarations -I$ROOT/include -I$TC/include -Iharness -Ikernels "$@" "$src" -o "$out" $LIBS
