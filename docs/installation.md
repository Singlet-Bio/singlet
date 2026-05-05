# Installation

## Python (PyPI)

```bash
# Core package (catalog browsing, .1pz I/O, AnnData integration)
pip install singlet

# With PyTorch support (sparse tensor dataloaders)
pip install singlet[torch]

# With GPU analysis (requires CUDA 12+)
pip install singlet[gpu]

# Everything
pip install singlet[all]
```

## R (GitHub / CRAN)

```r
# From GitHub (recommended for latest)
remotes::install_github("Singlet-Bio/singlet", subdir = "r")

# Requirements: C++17 compiler, libzstd >= 1.4
# Optional: CUDA 12+ for GPU functions
```

## C++ (CMake)

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet

# Header-only usage (just copy include/singlet/)
cmake -B build
cmake --install build --prefix /usr/local

# In your CMakeLists.txt:
find_package(Singlet REQUIRED COMPONENTS pz fq)
target_link_libraries(myapp Singlet::pz)
```

## Building the pipeline binary

```bash
cmake -B build \
    -DSINGLET_BUILD_PIPELINE=ON \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# Binary: build/src/pipeline/singlify
```

## Dependencies

| Component | Required | Optional |
|-----------|----------|----------|
| Python core | numpy, scipy, pandas, anndata, requests | torch, cupy, zarr |
| R package | Rcpp, Matrix, C++17, libzstd | CUDA 12+, SingleCellExperiment, Seurat |
| C++ headers | zstd, zlib | htslib (pipeline), CUDA (GPU) |
| Pipeline binary | htslib, zstd, zlib, OpenMP, ncbi-vdb | LZ4 |
