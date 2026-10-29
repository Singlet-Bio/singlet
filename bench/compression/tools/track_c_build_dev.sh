#!/usr/bin/env bash
# Build a harness source to a private exe so long runs of other binaries are not disturbed.
# Usage: bash tools/track_c_build_dev.sh [SRC=harness/track_c_eval.cpp] [OUT=bin/track_c_dev.exe] [extra g++ flags]
cd "$(dirname "$0")/.."
SRC=${1:-harness/track_c_eval.cpp}; OUT=${2:-bin/track_c_dev.exe}; shift 2 2>/dev/null
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
g++ -std=c++20 -O3 -march=native -mtune=native -DNDEBUG -w -I$ROOT/include -I$TC/include -Iharness -Ikernels "$@" $SRC -o $OUT $LIBS
