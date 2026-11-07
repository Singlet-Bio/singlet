#!/usr/bin/env bash
# Build the GPU engine binary (singlet --engine gpu) from a git ref of the public repository, with the recipe Clipper uses:
# a conda toolchain (the pbench env of bench/pipeline/setup_env.sbatch, reduced to the build dependencies) inside the CUDA 12.4.1 devel
# Ubuntu 22.04 image, -march=native replaced by -march=x86-64-v3, the bam_cat.c declaration, and the cmake line of
# scripts/engine_gpu/build_clipper.sbatch. The Dockerfile beside this file runs it; on a container platform without Docker, run it as
# root inside nvidia/cuda:12.4.1-devel-ubuntu22.04.
#
#   SINGLET_REPO  repository URL (default https://github.com/Singlet-Bio/singlet)
#   SINGLET_REF   branch, tag or commit to build (default gpu/frontier)
#   PREFIX        install root (default /opt/singlet): bin/singlet, whitelists/, src/, env/, COMMIT, build-info.txt
#   CUDA_ARCHS    CMAKE_CUDA_ARCHITECTURES (default "80;90": A100 and H100)
#   JOBS          parallel compile jobs (default: all cores)
#   CUDA_HOME     CUDA toolkit with nvcc (default /usr/local/cuda)
#   GCC_SPEC      conda version spec for the compilers, e.g. "=13" (default empty: the newest, as on Clipper, gcc 16 there)
set -euo pipefail
SINGLET_REPO=${SINGLET_REPO:-https://github.com/Singlet-Bio/singlet}
SINGLET_REF=${SINGLET_REF:-gpu/frontier}
PREFIX=${PREFIX:-/opt/singlet}
CUDA_ARCHS=${CUDA_ARCHS:-80;90}
JOBS=${JOBS:-$(nproc)}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
GCC_SPEC=${GCC_SPEC:-}
CONDA_DIR=$PREFIX/conda; ENV=$PREFIX/env; S=$PREFIX/src; B=${BUILD_DIR:-/tmp/singlet-build}
mkdir -p "$PREFIX/bin"
log() { echo "[build $(date +%H:%M:%S)] $*"; }

# System packages (only when run as root on a bare CUDA image; the Dockerfile has installed them already)
if [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  missing=""
  for t in git curl python3 /usr/bin/time flock; do command -v $t >/dev/null || missing=1; done
  if [ -n "$missing" ]; then
    log "apt: git curl ca-certificates time procps python3"
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      git curl ca-certificates time procps python3 util-linux pciutils >/dev/null
  fi
fi
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "no nvcc at $CUDA_HOME/bin/nvcc (set CUDA_HOME, or use the CUDA devel image)"; exit 2; }

# Conda toolchain. Package list: bench/pipeline/setup_env.sbatch (the pbench env) without the benchmark's comparison tools,
# plus its second install line (htslib.pc Requires.private). STAR 2.7.11b is kept: fetch.sh builds the genome index with it.
if [ ! -x "$CONDA_DIR/bin/mamba" ]; then
  log "Miniforge into $CONDA_DIR"
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR" >/dev/null; rm -f /tmp/miniforge.sh
fi
if [ ! -f "$ENV/.ok" ]; then
  log "conda env $ENV"
  rm -rf "$ENV"
  "$CONDA_DIR/bin/mamba" create -y -q -p "$ENV" -c conda-forge -c bioconda --strict-channel-priority \
    python=3.12 star=2.7.11b htslib ncbi-vdb hdf5 zstd zlib cmake make pkg-config \
    "gxx_linux-64$GCC_SPEC" "gcc_linux-64$GCC_SPEC" sysroot_linux-64 pigz
  "$CONDA_DIR/bin/mamba" install -y -q -p "$ENV" -c conda-forge liblzma-devel libdeflate bzip2 libcurl
  "$ENV/bin/pkg-config" --exists htslib || { echo "htslib.pc missing from $ENV"; exit 3; }
  touch "$ENV/.ok"
fi

# Source at the requested ref
if [ ! -d "$S/.git" ]; then git clone -q --filter=blob:none "$SINGLET_REPO" "$S"; fi
if git -C "$S" fetch -q origin "$SINGLET_REF" 2>/dev/null; then git -C "$S" checkout -q --detach FETCH_HEAD
else git -C "$S" fetch -q --tags origin; git -C "$S" checkout -q --detach "$SINGLET_REF"; fi
git -C "$S" reset -q --hard; git -C "$S" clean -qfdx
COMMIT=$(git -C "$S" rev-parse HEAD); log "source $SINGLET_REPO $SINGLET_REF = $COMMIT"

# The two source edits of Clipper's build (scripts/engine_gpu/build_clipper.sbatch): a portable ISA level instead of the build host's,
# and the C++ declaration of htslib's sam_hdr_destroy for STAR's bam_cat.c (compiled as C++)
grep -rl --include=CMakeLists.txt -e "-march=native" "$S" | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
grep -q "sam_hdr_destroy(bam_hdr_t" "$S/src/star/bam_cat.c" || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' "$S/src/star/bam_cat.c"

export PATH=$ENV/bin:$CUDA_HOME/bin:$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
# the conda libstdc++ before the CUDA image's system one (as on Clipper); the rpath lets the binary run without LD_LIBRARY_PATH
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
log "$($CXX --version | head -1); $(nvcc --version | tail -1)"
rm -rf "$B"; mkdir -p "$B"
cmake -S "$S" -B "$B" -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT="$COMMIT" -DSINGLET_ENGINE_GPU=ON \
  -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" > "$B/configure.log" 2>&1 \
  || { tail -40 "$B/configure.log"; exit 1; }
grep -i "GPU engine" "$B/configure.log" || true
T0=$(date +%s)
cmake --build "$B" --target singlet -j "$JOBS" > "$B/build.log" 2>&1 || { grep -m30 -B3 -A8 "error" "$B/build.log" || tail -40 "$B/build.log"; exit 1; }
log "compiled in $(( $(date +%s) - T0 )) s"

cp "$B/src/pipeline/singlet" "$PREFIX/bin/singlet.new" && mv "$PREFIX/bin/singlet.new" "$PREFIX/bin/singlet"
echo "$COMMIT" > "$PREFIX/COMMIT"
# the binary looks for <bin>/../whitelists; the 10x lists are in the repository (python/singlet/data/whitelists)
rm -rf "$PREFIX/whitelists"; cp -r "$S/python/singlet/data/whitelists" "$PREFIX/whitelists"
{ echo "repo $SINGLET_REPO ref $SINGLET_REF commit $COMMIT built $(date -Is) on $(hostname)"
  echo "cuda_archs $CUDA_ARCHS"; $CXX --version | head -1; nvcc --version | tail -1; cmake --version | head -1
  echo "htslib $(pkg-config --modversion htslib)"; echo "sha256 $(sha256sum "$PREFIX/bin/singlet" | cut -d' ' -f1)"
  "$ENV/bin/STAR" --version 2>/dev/null | sed 's/^/STAR /'
  echo "--- conda packages"; "$CONDA_DIR/bin/mamba" list -p "$ENV" 2>/dev/null | grep -v '^#'; } > "$PREFIX/build-info.txt"
grep -aqF "$COMMIT" "$PREFIX/bin/singlet" && log "commit $COMMIT is embedded in the binary" || log "WARNING: commit not found inside the binary"
ldd "$PREFIX/bin/singlet" | grep -iE "not found" && { echo "unresolved libraries"; exit 1; } || true
rm -rf "$B"; "$CONDA_DIR/bin/mamba" clean -afy -q >/dev/null 2>&1 || true
log "done: $PREFIX/bin/singlet ($COMMIT)"
