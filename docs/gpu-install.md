# Install

singlet-gpu is a header-only C++20 library with optional Python (pybind11) and R (Rcpp) wrappers. The C++ library has no runtime dependencies beyond CUDA itself.

## Supported platforms

| Axis | Supported |
|---|---|
| CUDA | 12.x |
| GPU arch | sm_70 (V100), sm_80 (A100), sm_90 (H100) |
| C++ | C++20 (gcc-toolset-13 + nvcc 12.x; clang ≥17) |
| Python | 3.10, 3.11, 3.12 |
| R | 4.3+ |
| OS | Linux x86_64 (RHEL 9 / Ubuntu 22.04 tested) |

Anything outside this matrix is best-effort.

## Required system dependencies

- CUDA 12.x toolkit (`nvcc`, cuBLAS, cuSPARSE, cuSOLVER, cuRAND, cuFFT, cuDNN, CUB).
- factornet (GPL-2.0) headers — fetched automatically via CMake FetchContent if not present locally.
- Eigen 3.4+ (transitively required by factornet CPU side; auto-discovered by our CMake from `FACTORNET_INCLUDE_DIR/../subprojects/eigen-3.4.0` if bundled).
- A C++20 compiler. **nvcc + g++ ≥ 13 (or clang ≥ 17) required.**

## ⚠ Library is nvcc-only

singlet-gpu is header-only, but several headers (notably `io/pz_device_loader.h` and the kernel modules) contain `<<<grid, block>>>` CUDA kernel launch syntax inline. **Consumers must compile their own translation units with `nvcc`**, not bare `g++`. Use `nvcc -ccbin g++ -x cu` for a `.cpp` source, or rename to `.cu`. Linking is unaffected (host-side, plain `g++` works).

This is why `examples/cpp_minimal/CMakeLists.txt` declares `LANGUAGES CXX CUDA` and sets `set_target_properties(... CUDA_ARCHITECTURES "70;80;90")`. Mirror this pattern in your own targets that include `<singlet-gpu/singlet_gpu.hpp>`.

## C++ install — CMake FetchContent (recommended for downstream projects)

```cmake
include(FetchContent)
FetchContent_Declare(
  singlet_gpu
  GIT_REPOSITORY https://github.com/Singlet-Bio/singlet.git
  GIT_TAG        v1.0.0  # pin a release
  SOURCE_SUBDIR  singlet-gpu
)
FetchContent_MakeAvailable(singlet_gpu)

target_link_libraries(your_target PRIVATE singlet-gpu::singlet-gpu)
```

The `singlet-gpu::singlet-gpu` INTERFACE target sets `-DFACTORNET_HAS_GPU=1`, the include path, and links the relevant CUDA libraries.

> **Status**: pre-1.0. The `v1.0.0` tag does not yet exist. Pin a commit hash in `GIT_TAG` until the first release is cut.

## C++ install — local build

```bash
git clone <repo> ~/singlet
cd ~/singlet/singlet-gpu

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++ \
  -DCMAKE_CUDA_ARCHITECTURES="70;80;90"

cmake --build build -j8
ctest --test-dir build
```

For Clipper-specific paths and the canonical sbatch build template, see [`../state/infrastructure.md`](../state/infrastructure.md).

## Python wrapper

```bash
pip install singlet-gpu
```

> **Status**: pre-1.0. Wheels are not yet on PyPI. Until then build from source:
>
> ```bash
> cd singlet-gpu/python && python -m pip install -e .
> ```

The package requires:
- `cupy-cuda12x ≥ 13.0` and a working CUDA installation matching the `nvcc` used for the C++ build.
- Python ≥ 3.10. (3.10, 3.11, 3.12, 3.13, 3.14 are all tested.)
- `numpy ≥ 1.24`. Optional: `anndata` for the `read_anndata` / `to_anndata` helpers.

### On HPC clusters with module systems

System Python is often older (e.g. 3.9 EOL on Clipper). Always load a supported Python via lmod / Spack, then bootstrap pip if missing, and **always invoke pip via `python -m pip`** (the bare `pip` binary often points at a different Python):

```bash
module load python/3.11.14         # or 3.10+ — see `module avail python`
python -m ensurepip --user --upgrade  # if pip not bundled
python -m pip install -e .
```

After install, also confirm CUDA libs are findable from the loaded Python:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## R wrapper

```r
remotes::install_github("Singlet-Bio/singlet", subdir = "singlet-gpu/r")
```

> **Status**: pre-1.0. The R package compiles against system CUDA at install time; ensure `nvcc` is on `PATH`.

## Verifying the install

```cpp
#include <singlet-gpu/singlet_gpu.hpp>
#include <iostream>

int main() {
    std::cout << "singlet-gpu " << singlet_gpu::version_major()
              << "." << singlet_gpu::version_minor()
              << "." << singlet_gpu::version_patch()
              << " (commit " << singlet_gpu::commit_sha() << ")\n";
}
```

```python
import singlet_gpu
print(singlet_gpu.__version__)
```

```r
library(singletGpu)
singletGpu::version()
```

If any of these fail at link time with `_M_replace_cold` or similar libstdc++ ABI errors, see [`../state/blockers.md`](../state/blockers.md) → INFRA-G008-LIBSTDCPP.
