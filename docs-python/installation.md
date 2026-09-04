# Installation

```bash
pip install singlet
```

## Optional extras

```bash
pip install "singlet[torch]"   # PyTorch DataLoaders
pip install "singlet[gpu]"     # GPU-accelerated analysis (CUDA 12)
pip install "singlet[zarr]"    # Zarr export/import
pip install "singlet[tiledb]"  # TileDB-SOMA export/import
pip install "singlet[mcp]"     # Model Context Protocol server for AI assistants
pip install "singlet[all]"     # Everything
```

Everything else — clustering, differential expression, trajectory
inference, plotting — works out of the box with just the base install.
Individual analysis functions that depend on an optional third-party
algorithm package (`leidenalg`, `umap-learn`, `harmonypy`, `phate`,
`palantir`, ...) fall back to a pure-Python or SciPy implementation when
that package isn't installed, rather than failing to import.

## From source

```bash
git clone https://github.com/Singlet-Bio/singlet.git
cd singlet
pip install -e .
```

Building the compiled `.1pz` codec extension needs a C++17 compiler and
`libzstd` (headers + library) available on the system — on Debian/Ubuntu,
`apt install libzstd-dev`; on macOS, `brew install zstd`.
