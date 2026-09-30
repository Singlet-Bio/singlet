# singlet

A single-cell genomics toolkit: process, load, and analyze public single-cell
RNA-seq data. It is the client for the [singlet.bio](https://singlet.bio)
catalog of re-processed GEO studies.

**One repository. Three languages.**

```bash
# Python — from GitHub until the first PyPI release (then: pip install singlet-bio)
pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"
```
```r
remotes::install_github("Singlet-Bio/singlet", subdir = "r")  # R
```
```cmake
find_package(Singlet REQUIRED)         # C++
```

The PyPI distribution is named **`singlet-bio`** (`singlet` on PyPI is an
unrelated project); the import name is still `singlet`. Once the first
release is published, the install is `pip install singlet-bio`.

Installing the Python package from source compiles a small C++ extension (the
`.1pz` codec), so you need a **C++17 compiler** and **zstd** headers + library
(`apt install build-essential libzstd-dev`, `brew install zstd`, or
`conda install -c conda-forge zstd`). Linux and macOS are supported; Windows
users should use WSL.

## What singlet does

| Layer | Capability |
|-------|-----------|
| **Catalog** | Search and load re-processed GEO studies from singlet.bio as AnnData — see the site for current numbers |
| **Format** | `.singlet` per-study bundles and the `.1pz` sparse-matrix codec |
| **Pipeline** | Raw SRA reads → aligned, deduplicated, annotated count matrices (C++ binary) |
| **Analysis** | PCA, neighbors, Leiden, UMAP, DE, batch correction — no scanpy required |
| **PyTorch** | Sparse DataLoaders over `.singlet` files for ML training |

## Quick Start

```python
import singlet

# Find studies in plain English (live search) → GSE accessions
singlet.find("human PBMC 10x")

# Load a study → AnnData (free download from data.singlet.bio, cached)
adata = singlet.load("GSE138867")

# Load one sample of it (downloads the parent study, keeps this sample's cells)
pbmc1 = singlet.load("GSM4120733")

# Load several studies into ONE AnnData (obs["dataset"] records each study)
adata = singlet.load(["GSE138867", "GSE146974"])

# Subset genes by symbol or Ensembl id (var_names stay Ensembl ids)
tcells = singlet.load("GSE138867", genes=["CD3E", "CD8A", "MS4A1"])

# Explore
singlet.describe(adata)                     # quick summary stats

# Preprocessing (no scanpy required)
singlet.filter_cells(adata, min_genes=200, inplace=True)
singlet.filter_genes(adata, min_cells=3, inplace=True)
singlet.normalize(adata)                    # library-size + log1p
singlet.highly_variable_genes(adata)        # feature selection

# Analysis
singlet.pca(adata)                          # dimensionality reduction
singlet.harmony(adata, "dataset")           # batch-correct across the two studies
singlet.neighbors(adata, use_rep="X_pca_harmony")
singlet.leiden(adata)                       # clustering
singlet.umap(adata)                         # 2D embedding

# Differential expression & visualization (plotting needs matplotlib)
singlet.rank_genes_groups(adata, "leiden")
singlet.plot_umap(adata, color="leiden")

# Export
singlet.to_h5ad(adata, "pbmc.h5ad")         # also to_zarr, to_mtx
```

Samples whose count matrix is empty are skipped with a warning and listed in
`adata.uns["skipped_samples"]`. Loom export is not supported from Python; use
`.h5ad`, `.zarr` or 10x MTX.

## Repository Layout

```
singlet/
├── include/singlet/     C++ headers (libsinglet — header-only)
│   ├── pz/              .1pz format codec
│   ├── fq/              .1fq encoded FASTQ codec
│   ├── pileup/          Streaming BAM pileup engine
│   └── gpu/             CUDA analysis kernels (experimental)
├── python/singlet/      Python package source
│   ├── io/              Format I/O (.1pz, h5ad, zarr)
│   ├── preprocessing/   Pipeline QC & preprocessing
│   ├── torch/           PyTorch integration
│   └── gpu/             GPU analysis wrappers (experimental)
├── r/                   R package
├── src/                 Compiled sources (pipeline binary, STAR aligner, GPU kernels)
├── tests/               Test suites (Python + C++ + R)
├── notebooks/           Jupyter tutorial notebooks
├── docs/                User documentation
└── papers/              Scientific manuscripts
```

## Installation

See [docs/installation.md](docs/installation.md) for full details. The Python
distribution is `singlet-bio` (after the first PyPI release:
`pip install singlet-bio`); the import name is `singlet`.

| Install | Command |
|---------|---------|
| Python (core) | `pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"` |
| Python + analysis extras | `pip install "singlet-bio[analysis] @ git+https://github.com/Singlet-Bio/singlet"` |
| Python + PyTorch | `pip install "singlet-bio[torch] @ git+https://github.com/Singlet-Bio/singlet"` |
| Python + local MCP server | `pip install "singlet-bio[mcp] @ git+https://github.com/Singlet-Bio/singlet"` |
| R | `remotes::install_github("Singlet-Bio/singlet", subdir = "r")` |
| C++ (CMake) | `find_package(Singlet COMPONENTS pz fq pileup)` |
| Pipeline binary | `cmake -B build -DSINGLET_BUILD_PIPELINE=ON` |

The GPU module (`singlet.gpu`) is experimental: `singlet-bio[gpu]` only installs
CuPy, and the CUDA extension it needs must be built from source on a CUDA 12
machine.

## MCP servers

There are two, and they are different:

- **Hosted:** `https://singlet.bio/mcp` — the live catalog, maintained by the
  site. Point an MCP client at the URL; nothing to install.
- **Local:** `singlet-mcp` (from `singlet-bio[mcp]`) — 12 tools over the offline
  catalog snapshot bundled with the package plus live search. The old command
  name `singlet` still starts it but is deprecated, because `singlet` is also
  the pipeline binary's name.

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01_load_and_explore](notebooks/01_load_and_explore.ipynb) | Full analysis pipeline |
| [quickstart](notebooks/quickstart.ipynb) | Catalog API |
| [gene_counting](notebooks/gene_counting.ipynb) | STARsolo equivalence |
| [1pz_format](notebooks/1pz_format.ipynb) | Format internals |
| [doublet_detection](notebooks/doublet_detection.ipynb) | UMI-based doublet detection |
| [rna_velocity](notebooks/rna_velocity.ipynb) | Spliced/unspliced for scVelo |
| [cell_calling](notebooks/cell_calling.ipynb) | EmptyDrops deviance testing |
| + more | QC, ancestry, sex, splicing, saturation, etc. |

## Building from Source

```bash
# Pipeline binary (C++ — requires htslib, zstd, ncbi-vdb, OpenMP; HDF5 optional)
source /opt/rh/gcc-toolset-13/enable  # RHEL/Rocky — need GCC 13+
export CONDA_PREFIX=/path/to/conda/env  # must have htslib, ncbi-vdb
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
cmake -B build -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# Binary: build/src/pipeline/singlet — set $SINGLET_BINARY to it (or put it on
# $PATH as singlet-pipeline) for singlet.run_pipeline / singlet-process.

# C++ tests
cmake -B build -DSINGLET_BUILD_TESTS=ON
cmake --build build -j$(nproc) && ctest --test-dir build -j$(nproc)

# Python tests
pip install -e ".[dev]"
pytest tests/python/

# Lint
make lint
```

## License

MIT. See [LICENSE](LICENSE). Vendored STAR sources under
`include/singlet/star/` keep their own licenses.
