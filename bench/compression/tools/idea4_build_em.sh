#!/usr/bin/env bash
cd "$(dirname "$0")/.."
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
mkdir -p bin
g++ -std=c++20 -O3 -march=native -fopenmp -shared -static-libgcc -static-libstdc++ tools/idea4_em.cpp -o bin/idea4_em.dll -Wl,-Bstatic -lgomp -lpthread -Wl,-Bdynamic 2>&1 | head -20
ls -la bin/idea4_em.dll
