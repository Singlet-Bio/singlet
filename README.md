# singlet

A unified single-cell genomics library: process, load, and analyze single-cell data at scale.

**One package. Three languages. Zero friction.**

```python
pip install singlet                    # Python
```
```r
remotes::install_github("Singlet-Bio/singlet", subdir = "r")  # R
```
```cmake
find_package(Singlet REQUIRED)         # C++
```

## What singlet does

| Layer | Capability |
|-------|-----------|
| **Format** | .1pz sparse matrix codec (13x compression, 4000+ MB/s decode) |
| **Pipeline** | Raw SRA reads → aligned, deduplicated, annotated .1pz files |
| **Analysis** | GPU-accelerated PCA, NMF, kNN, Leiden, UMAP, DE, integration |
| **Atlas** | Browse and load 4,600+ processed single-cell samples |
| **PyTorch** | Zero-copy sparse DataLoaders for ML training |

## Quick Start

```python
import singlet

# Browse the atlas
singlet.catalog()
singlet.info("GSE264667")

# Load a sample (returns AnnData)
adata = singlet.load("GSM1234567")

# File I/O
adata = singlet.read_1pz("counts.1pz")
singlet.write_1pz(adata, "output.1pz")

# GPU analysis (pip install singlet[gpu])
from singlet import gpu
from singlet.gpu import preprocess as sgpp
sgpp.normalize_total(adata)
sgpp.log1p(adata)
gpu.reduce.pca(adata, n_components=50)
gpu.pp.neighbors(adata)
gpu.tools.leiden(adata)
gpu.tools.umap(adata)

# PyTorch training (pip install singlet[torch])
from singlet.torch import OnePZDataset, DataLoader
dataset = OnePZDataset("counts.1pz")
loader = DataLoader(dataset, batch_size=512)
```

## Repository Layout

```
singlet/
├── include/singlet/     C++ headers (libsinglet — header-only)
│   ├── pz/              .1pz format codec
│   ├── fq/              .1fq encoded FASTQ codec
│   ├── pileup/          Streaming BAM pileup engine
│   └── gpu/             CUDA analysis kernels
├── python/singlet/      Python package source
│   ├── io/              Format I/O (.1pz, h5ad, zarr)
│   ├── preprocessing/   Pipeline QC & preprocessing
│   ├── torch/           PyTorch integration
│   └── gpu/             GPU analysis wrappers
├── r/                   R package (CRAN-ready)
├── src/                 Compiled sources (pipeline binary, STAR aligner, GPU kernels)
├── tests/               Unified test suite (Python + C++ + R)
├── notebooks/           Jupyter tutorial notebooks
├── docs/                User documentation
└── papers/              Scientific manuscripts
```

## Installation

See [docs/installation.md](docs/installation.md) for full details.

| Install | Command |
|---------|---------|
| Python (core) | `pip install singlet` |
| Python + GPU | `pip install singlet[gpu]` |
| Python + PyTorch | `pip install singlet[torch]` |
| Python (everything) | `pip install singlet[all]` |
| R | `remotes::install_github("Singlet-Bio/singlet", subdir = "r")` |
| C++ (CMake) | `find_package(Singlet COMPONENTS pz fq pileup)` |
| Pipeline binary | `cmake -B build -DSINGLET_BUILD_PIPELINE=ON` |

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01_load_and_explore](notebooks/01_load_and_explore.ipynb) | Full analysis pipeline |
| [02_gpu_analysis](notebooks/02_gpu_analysis.ipynb) | GPU-accelerated workflows |
| [quickstart](notebooks/quickstart.ipynb) | Catalog API |
| [gene_counting](notebooks/gene_counting.ipynb) | STARsolo equivalence (r=0.9995) |
| [1pz_format](notebooks/1pz_format.ipynb) | Format internals |
| [doublet_detection](notebooks/doublet_detection.ipynb) | UMI-based doublet detection |
| [rna_velocity](notebooks/rna_velocity.ipynb) | Spliced/unspliced for scVelo |
| [cell_calling](notebooks/cell_calling.ipynb) | EmptyDrops deviance testing |
| + 10 more | QC, ancestry, sex, splicing, saturation, etc. |

## Building from Source

```bash
# Pipeline binary (C++ — requires htslib, zstd, ncbi-vdb, OpenMP; HDF5 optional)
source /opt/rh/gcc-toolset-13/enable  # RHEL/Rocky — need GCC 13+
export CONDA_PREFIX=/path/to/conda/env  # must have htslib, ncbi-vdb
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
cmake -B build -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# GPU library (requires CUDA 12+)
cmake -B build -DSINGLET_BUILD_GPU=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run C++ tests (100 unit tests — ~72s)
cmake -B build -DSINGLET_BUILD_TESTS=ON
cmake --build build -j$(nproc) && ctest --test-dir build -j$(nproc)

# Run Python tests (600 tests)
pip install -e ".[dev]"
pytest tests/python/

# Lint
make lint
```

## License

MIT (core library) — GPU kernels are GPL-2.0. See [LICENSE](LICENSE).
