# GPU Installation

The singlet GPU module is a header-only C++20/CUDA library with Python (pybind11) and R (Rcpp) wrappers. It lives within the unified singlet repository.

## Supported platforms

| Axis | Supported |
|---|---|
| CUDA | 12.x |
| GPU arch | sm_70 (V100), sm_80 (A100), sm_90 (H100) |
| C++ | C++20 (gcc-toolset-13 + nvcc 12.x; clang >= 17) |
| Python | 3.10, 3.11, 3.12 |
| R | 4.2+ |
| OS | Linux x86_64 (RHEL 9 / Ubuntu 22.04 tested) |

## Python

```bash
pip install singlet[gpu]
```

Requires:
- `cupy-cuda12x >= 13.0` and a working CUDA installation
- Python >= 3.10
- `numpy >= 1.24`

### Build from source

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet
pip install -e ".[gpu]"
```

## R

```r
remotes::install_github("Singlet-Bio/singlet", subdir = "r")
```

Ensure `nvcc` is on `PATH` for GPU support at install time.

## C++ (CMake)

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet

cmake -S . -B build \
  -DSINGLET_BUILD_GPU=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="70;80;90"

cmake --build build -j$(nproc)
```

### CMake FetchContent (downstream projects)

```cmake
include(FetchContent)
FetchContent_Declare(
  singlet
  GIT_REPOSITORY https://github.com/Singlet-Bio/singlet.git
  GIT_TAG        main
)
FetchContent_MakeAvailable(singlet)

target_link_libraries(your_target PRIVATE Singlet::gpu)
```

## Required system dependencies

- CUDA 12.x toolkit (nvcc, cuBLAS, cuSPARSE, cuSOLVER, cuRAND)
- A C++20 compiler (nvcc + g++ >= 13, or clang >= 17)

## HPC clusters

System Python is often older. Always load a supported Python via lmod/Spack:

```bash
module load python/3.11
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip install singlet[gpu]
```

## Verifying the install

```python
from singlet.gpu import device_count
print(f"GPUs available: {device_count()}")
```

```r
library(singlet)
singlet::has_gpu()
```
