# Installation

## Python

Wheels are not published to PyPI yet. Install from GitHub (after the first
release: `pip install singlet-bio`). The distribution is named `singlet-bio`
because `singlet` on PyPI is an unrelated project; you still `import singlet`.

```bash
# Core package: find/load studies as AnnData, .1pz I/O, catalog snapshot
pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"

# Optional extras use the same URL
pip install "singlet-bio[analysis] @ git+https://github.com/Singlet-Bio/singlet"  # matplotlib, scanpy, statsmodels, igraph, leidenalg
pip install "singlet-bio[torch] @ git+https://github.com/Singlet-Bio/singlet"     # PyTorch DataLoaders
pip install "singlet-bio[zarr] @ git+https://github.com/Singlet-Bio/singlet"      # Zarr export/import
pip install "singlet-bio[mcp] @ git+https://github.com/Singlet-Bio/singlet"       # local MCP server (singlet-mcp)
```

Installing from source compiles the `.1pz` codec extension, which needs:

- a **C++17 compiler** (GCC ≥ 7, Clang ≥ 5, or Xcode command-line tools), and
- **zstd** headers and library: `apt install build-essential libzstd-dev`
  (Debian/Ubuntu), `dnf install gcc-c++ libzstd-devel` (Fedora/RHEL),
  `brew install zstd` (macOS), or `conda install -c conda-forge zstd`.

Linux and macOS are supported. On Windows, use WSL.

Check the install:

```bash
python -m singlet        # prints the version and live catalog totals
```

The GPU module (`singlet.gpu`) is experimental. `singlet-bio[gpu]` installs only
CuPy; the CUDA extension (`singlet.gpu._core`) must be built from source on a
CUDA 12 machine and is not produced by `pip install`.

## R (GitHub)

```r
remotes::install_github("Singlet-Bio/singlet", subdir = "r")

# Requirements: C++17 compiler, libzstd >= 1.4
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
# Binary: build/src/pipeline/singlet
```

The Python wrapper (`singlet.run_pipeline`, `singlet-process`) finds the
binary through `$SINGLET_BINARY`, then `singlet-pipeline` on `$PATH`, then a
`singlet` on `$PATH` that is not a Python script. (The Python package installs
a deprecated `singlet` command for its MCP server; it is skipped.)

## Dependencies

| Component | Required | Optional |
|-----------|----------|----------|
| Python core | numpy, scipy, pandas, anndata, requests, pyarrow, scikit-learn, zstandard, tqdm; C++17 + zstd to build | torch, zarr, tiledbsoma, mcp, matplotlib, scanpy |
| R package | Rcpp, Matrix, C++17, libzstd | SingleCellExperiment, Seurat |
| C++ headers | zstd, zlib | htslib (pipeline), CUDA (GPU) |
| Pipeline binary | htslib, zstd, zlib, OpenMP, ncbi-vdb | LZ4, HDF5 |
