# Installation

`pip install singlet` installs a wheel with the compiled codec (Linux x86_64/aarch64, macOS Intel/Apple silicon,
Windows). The commands below install the development version from GitHub, which builds the codec from source.

```bash
pip install "singlet @ git+https://github.com/Singlet-Bio/singlet"
```

This compiles the package's `.1pz` codec extension, so it needs a **C++17
compiler** and **zstd** (headers + library) on the system — on
Debian/Ubuntu, `apt install build-essential libzstd-dev`; on macOS,
`brew install zstd`; with conda, `conda install -c conda-forge zstd`.
On Windows, install Visual Studio Build Tools ("Desktop development with C++") and zstd
(`vcpkg install zstd:x64-windows`), or use the Windows wheels once a release is published.
The R package builds and is tested on Windows with Rtools.

## Optional extras

Extras use the same URL:

```bash
pip install "singlet[analysis] @ git+https://github.com/Singlet-Bio/singlet"  # matplotlib, scanpy, statsmodels, igraph, leidenalg
pip install "singlet[torch] @ git+https://github.com/Singlet-Bio/singlet"     # PyTorch DataLoaders
pip install "singlet[zarr] @ git+https://github.com/Singlet-Bio/singlet"      # Zarr export/import
pip install "singlet[tiledb] @ git+https://github.com/Singlet-Bio/singlet"    # TileDB-SOMA export/import
pip install "singlet[mcp] @ git+https://github.com/Singlet-Bio/singlet"       # local MCP server (singlet-mcp)
```

Clustering, differential expression, trajectory inference and the other
analysis functions work with the base install. Functions that can use an
optional algorithm package (`leidenalg`, `umap-learn`, `harmonypy`, `phate`,
`palantir`, ...) fall back to a pure-Python or SciPy implementation when that
package isn't installed; plotting functions need `matplotlib`.

The GPU module (`singlet.gpu`) is experimental: `singlet[gpu]` installs only
CuPy, and its CUDA extension must be built from source on a CUDA 12 machine.

## From a clone

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet
pip install -e ".[dev]"   # editable install with test and lint tools
pytest tests/python/
```
