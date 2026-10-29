#!/usr/bin/env bash
# Build the phase 1d harness. NOT COMPILED with nvcc yet (written on a laptop without it); the host check below was compiled with
# g++ 13.2 and passes. Run from the repo root or from here.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
inc="$here/../../src/engine_gpu"

# 1. no GPU needed: the kernel stage against the Python model on the host (g++ >= 9, C++17)
g++ -std=c++17 -O1 -I"$inc" "$here/paired_unit_host.cpp" -o "$here/paired_unit_host"
"$here/paired_unit_host" "$here/fixtures/cases.txt"

# 2. the GPU harness (H100 = sm_90; CUDA_ARCH=sm_80 for A100)
if command -v nvcc >/dev/null 2>&1; then
    nvcc -std=c++17 -O3 -lineinfo -arch="${CUDA_ARCH:-sm_90}" -I"$inc" \
         "$here/paired_microbench.cu" -o "$here/paired_microbench"
    "$here/paired_microbench" --unit "$here/fixtures/cases.txt"
else
    echo "nvcc not found: skipped the GPU harness" >&2
fi
