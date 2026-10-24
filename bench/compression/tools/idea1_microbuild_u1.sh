#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
g++ -std=c++20 -O3 -march=native -mtune=native -DNDEBUG -DIDEA1_UNROLL=1 -Wall -Wno-unused-function -Ikernels harness/idea1_micro.cpp -o bin/idea1_micro_u1.exe -static && echo "[micro-u1] built"
