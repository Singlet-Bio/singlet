#!/usr/bin/env bash
# Build the compression-study harness with the Rtools 4.4 MinGW toolchain (GCC 13, static zstd/lzma/brotli/lz4/zlib/bz2).
# Usage: bash build.sh [target ...]     (default: all .cpp in harness/ and kernels/ that define main())
set -uo pipefail
cd "$(dirname "$0")"
TC=/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
ROOT="$(cd ../.. && pwd)"
mkdir -p bin
CXX=g++
CXXFLAGS="-std=c++20 -O3 -march=native -mtune=native -DNDEBUG -Wall -Wno-unused-function -I$ROOT/include -I$TC/include -Iharness -Ikernels"
LIBS="$TC/lib/libzstd.a $TC/lib/liblzma.a $TC/lib/libbrotlienc.a $TC/lib/libbrotlidec.a $TC/lib/libbrotlicommon.a $TC/lib/liblz4.a $TC/lib/libz.a $TC/lib/libbz2.a -lpthread -static"
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  targets=()
  for f in harness/*.cpp kernels/*.cpp; do
    [ -f "$f" ] || continue
    grep -q "int main" "$f" && targets+=("$f")
  done
fi
for src in "${targets[@]}"; do
  out="bin/$(basename "${src%.cpp}").exe"
  echo "[build] $src -> $out"
  if ! $CXX $CXXFLAGS "$src" -o "$out" $LIBS; then echo "[build] FAILED: $src"; fail=1; fi
done
echo "[build] done"
exit ${fail:-0}
