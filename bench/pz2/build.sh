#!/usr/bin/env bash
# Build a pz2 test / bench with the Rtools 4.4 GCC (see memory: laptop C++ toolchain).  NO -march flag on purpose: the SIMD
# paths must come from runtime dispatch.
# Usage: bash bench/pz2/build.sh <src.cpp> [<out-name>] [extra g++ flags...]   (output: bench/pz2/bin/<out-name>.exe)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
src="$1"; shift
out="$(basename "${src%.cpp}")"
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then out="$1"; shift; fi
mkdir -p "$HERE/bin"
g++ -std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -I"$ROOT/include" -I"$ROOT/tests/cpp" -I"$TC/include" "$@" \
    "$src" -o "$HERE/bin/$out.exe" "$TC/lib/libzstd.a" -lpthread -static
