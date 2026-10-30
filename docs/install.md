# GPU module — install status (experimental)

The GPU analysis code that used to be the standalone `singlet-gpu` project now
lives inside this repository:

| Piece | Location |
|---|---|
| CUDA kernels (header-only C++) | `include/singlet/gpu/`, `src/gpu/` |
| Python wrappers | `python/singlet/gpu/` (imported as `singlet.gpu`) |
| Python extension source | `src/bindings/python/_gpu_core.cpp` (module `singlet.gpu._core`) |

There is **no separate `singlet-gpu` package**: no `pip install singlet-gpu`
wheel, no `singletGpu` R package and no `singlet-gpu/` subdirectory to
build from. Older instructions that mention those are obsolete.

For everyday use you do not need any of this — loading studies
(`singlet.load`), search (`singlet.find`) and the CPU analysis functions work
with the normal install:

```bash
pip install "singlet @ git+https://github.com/Singlet-Bio/singlet"
```

## What `singlet[gpu]` does today

`pip install "singlet[gpu] @ git+https://github.com/Singlet-Bio/singlet"`
installs **CuPy only** (`cupy-cuda12x`). It does not compile the CUDA
extension, so `import singlet.gpu` warns that `singlet.gpu._core` could not be
imported and the functions that need it are unavailable. `pip install` does
not build `_gpu_core.cpp`, and no other packaged build target for it exists
yet.

## Building the C++ GPU library

On a machine with the CUDA 12 toolkit (`nvcc`, cuBLAS, cuSPARSE, cuSOLVER,
cuRAND) and GCC ≥ 13 (or Clang ≥ 17):

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet
cmake -B build-gpu \
  -DSINGLET_BUILD_GPU=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="70;80;90"
cmake --build build-gpu -j8
```

Headers that contain `<<<grid, block>>>` launches must be compiled by `nvcc`
in consuming projects (use `nvcc -x cu`, or `.cu` sources with
`LANGUAGES CXX CUDA` in CMake).

| Axis | Tested |
|---|---|
| CUDA | 12.x |
| GPU arch | sm_70 (V100), sm_80 (A100), sm_90 (H100) |
| OS | Linux x86_64 |
