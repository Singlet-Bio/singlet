#!/usr/bin/env bash
# Build the w4_reader prototype DLL (own target only; same toolchain as build.sh).
set -uo pipefail
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
mkdir -p bin
g++ -std=c++20 -O3 -march=native -DNDEBUG -Wall -Wno-unused-function -Wno-deprecated-declarations \
  -I"$ROOT/include" -I"$TC/include" -Ikernels -shared -o bin/w4_reader_dll.dll harness/w4_reader_dll.cpp \
  "$TC/lib/libzstd.a" -static -static-libgcc -static-libstdc++ -lpthread
echo "[build] exit $?"
