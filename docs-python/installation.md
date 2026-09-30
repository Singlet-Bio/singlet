# Installation

Wheels are not published to PyPI yet. Install from GitHub (after the first
release: `pip install singlet-bio`). The distribution is named `singlet-bio`
because `singlet` on PyPI is an unrelated project; you still `import singlet`.

```bash
pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"
```

This compiles the package's `.1pz` codec extension, so it needs a **C++17
compiler** and **zstd** (headers + library) on the system — on
Debian/Ubuntu, `apt install build-essential libzstd-dev`; on macOS,
`brew install zstd`; with conda, `conda install -c conda-forge zstd`.
Linux and macOS are supported; on Windows use WSL.

## Optional extras

Extras use the same URL:

```bash
pip install "singlet-bio[analysis] @ git+https://github.com/Singlet-Bio/singlet"  # matplotlib, scanpy, statsmodels, igraph, leidenalg
pip install "singlet-bio[torch] @ git+https://github.com/Singlet-Bio/singlet"     # PyTorch DataLoaders
pip install "singlet-bio[zarr] @ git+https://github.com/Singlet-Bio/singlet"      # Zarr export/import
pip install "singlet-bio[tiledb] @ git+https://github.com/Singlet-Bio/singlet"    # TileDB-SOMA export/import
pip install "singlet-bio[mcp] @ git+https://github.com/Singlet-Bio/singlet"       # local MCP server (singlet-mcp)
```

Clustering, differential expression, trajectory inference and the other
analysis functions work with the base install. Functions that can use an
optional algorithm package (`leidenalg`, `umap-learn`, `harmonypy`, `phate`,
`palantir`, ...) fall back to a pure-Python or SciPy implementation when that
package isn't installed; plotting functions need `matplotlib`.

The GPU module (`singlet.gpu`) is experimental: `singlet-bio[gpu]` installs only
CuPy, and its CUDA extension must be built from source on a CUDA 12 machine.

## From a clone

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet
pip install -e ".[dev]"   # editable install with test and lint tools
pytest tests/python/
```
