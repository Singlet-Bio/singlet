# singlet

**Python client for SingletDB — the world's largest uniformly processed single-cell database.**

19,790+ datasets • 24 species • ~10× compression • Zero-copy PyTorch sparse tensors

## Install

```bash
pip install singlet                # Core + AnnData
pip install singlet[torch]         # + PyTorch GPU support
pip install singlet[all]           # + zarr, TileDB-SOMA, torch
```

## Quick Start

```python
import singlet

# Browse the catalog (free, works offline)
df = singlet.catalog(search="lung")

# Load a dataset → AnnData
adata = singlet.load("GSE136831")
print(adata)

# PyTorch sparse tensor (zero-copy, GPU-ready)
from singlet.torch import to_sparse_csr
tensor = to_sparse_csr("~/.singlet/data/GSE136831.1pz", device="cuda")

# Convert formats
singlet.to_h5ad(adata, "output.h5ad")
singlet.to_zarr(adata, "output.zarr")
```

## Features

| Feature | Details |
|---------|---------|
| **Catalog** | Browse 19,790 datasets by species, tissue, modality |
| **Compression** | SinglePress .1pz format — ~13× vs raw CSC |
| **PyTorch** | Zero-copy sparse CSR/COO tensors, GPU via cuSPARSE |
| **Formats** | HDF5, Zarr, TileDB-SOMA, MTX, CSC sparse |
| **Queries** | Cross-atlas structured query + semantic search |
| **NMF** | Server-side NMF projection with biological annotations |
| **Preprocessing** | Full FASTQ→.1pz pipeline (simpleaf + piscem) |

## Architecture

```
singlet/
├── include/singlepress/
│   └── singlepress.h          # Header-only C++ compression library
├── src/
│   └── _singlepress.cpp       # pybind11 bindings (PyTorch zero-copy)
├── singlet/
│   ├── __init__.py             # Public API
│   ├── _io.py                  # .1pz/.spz read/write → AnnData
│   ├── _catalog.py             # Dataset catalog (Zenodo parquet)
│   ├── _loader.py              # Download + load orchestration
│   ├── _auth.py                # API key management
│   ├── _query.py               # Cross-atlas queries
│   ├── nmf.py                  # NMF projection + annotation
│   ├── torch.py                # PyTorch sparse tensors + DataLoaders
│   ├── convert.py              # Format interop (h5ad, zarr, tiledb, mtx)
│   └── preprocessing/          # FASTQ → .1pz pipeline
│       ├── _download.py        # ENA/SRA parallel downloads
│       ├── _detect.py          # Protocol detection
│       ├── _quantify.py        # simpleaf quantification
│       ├── _qc.py              # Quality control
│       ├── _species.py         # Species reference configs
│       └── _export.py          # Count matrix → .1pz export
├── tests/
├── docs/
└── pyproject.toml
```

## License

MIT
