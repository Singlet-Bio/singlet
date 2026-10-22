#!/usr/bin/env bash
# Build singlet._pz2 (and optionally singlet._pz) with the Rtools 4.4 MinGW GCC toolchain, for machines without MSVC.
# The result is a .pyd that imports only python3xx.dll and the UCRT (libstdc++, libgcc, winpthread and zstd are linked in).
#
#   bash src/bindings/python/build_pz2_mingw.sh            # -> python/singlet/_pz2.cp311-win_amd64.pyd
#   bash src/bindings/python/build_pz2_mingw.sh --pz       # also rebuild _pz (legacy reader)
#   PYTHON=python3.12 bash src/bindings/python/build_pz2_mingw.sh
#   CXXFLAGS_EXTRA="-O3" bash ...                          # extra flags (default optimisation is -O2, no -march: the SIMD paths
#                                                          # are selected at run time by cpuid)
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
TC=${TC:-/c/RBuildTools/4.4/x86_64-w64-mingw32.static.posix}
export PATH="$TC/bin:/c/RBuildTools/4.4/usr/bin:$PATH"
PY=${PYTHON:-python}

# one value per line (paths may contain spaces); Windows Python prints CRLF
mapfile -t INFO < <($PY -P - <<'EOF' | tr -d '\r'
import sys, sysconfig, os, pybind11
inc = sysconfig.get_paths()["include"]
ext = sysconfig.get_config_var("EXT_SUFFIX")
ver = "%d%d" % sys.version_info[:2]
cands = [os.path.join(sys.base_prefix, "python%s.dll" % ver), os.path.join(os.path.dirname(sys.executable), "python%s.dll" % ver)]
dll = next((c for c in cands if os.path.exists(c)), cands[0])
for v in (inc, ext, dll, pybind11.get_include()):
    print(v.replace("\\", "/"))
EOF
)
PYINC=${INFO[0]}; EXT=${INFO[1]}; PYDLL=${INFO[2]}; PYBIND=${INFO[3]}
OUTDIR="$ROOT/python/singlet"
CXXFLAGS=(-std=c++17 -O2 -DNDEBUG -fvisibility=hidden -Wall -Wno-unused-function -Wno-attributes -DMS_WIN64
          "-I$ROOT/include" "-I$ROOT/src/bindings/common" "-I$PYBIND" "-I$PYINC" "-I$TC/include")
if [ -n "${CXXFLAGS_EXTRA:-}" ]; then read -r -a EXTRA <<< "$CXXFLAGS_EXTRA"; CXXFLAGS+=("${EXTRA[@]}"); fi
LINK=(-shared -static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -Wl,-Bdynamic "$TC/lib/libzstd.a")

build_one() {
  local src="$1" name="$2"
  echo "[build] $src -> $OUTDIR/$name$EXT"
  g++ "${CXXFLAGS[@]}" "$ROOT/$src" -o "$OUTDIR/$name$EXT" "${LINK[@]}" "$PYDLL"
}
build_one src/bindings/python/_pz2.cpp _pz2
if [ "${1:-}" = "--pz" ]; then build_one src/bindings/python/_pz.cpp _pz; fi
echo "[build] done"
