# Installation

```bash
pip install singlet-bio        # the importable package is still `import singlet`
```

The distribution name on PyPI is `singlet-bio` — `singlet` was already taken
by an unrelated project — but the package you `import` stays `singlet`.

## Optional extras

```bash
pip install "singlet-bio[torch]"   # PyTorch DataLoaders
pip install "singlet-bio[gpu]"     # GPU-accelerated analysis (CUDA 12)
pip install "singlet-bio[zarr]"    # Zarr export/import
pip install "singlet-bio[tiledb]"  # TileDB-SOMA export/import
pip install "singlet-bio[mcp]"     # Model Context Protocol server for AI assistants
pip install "singlet-bio[all]"     # Everything
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
