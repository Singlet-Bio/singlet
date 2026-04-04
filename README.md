# singlet

**Python client for the Singlet single-cell atlas.**

3,309 datasets • 354M cells • 24 species • ~13× compression • Embedded metadata • PyTorch GPU tensors

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
human = singlet.datasets(organism="Homo sapiens", min_cells=100_000)

# Load a dataset → AnnData (with embedded obs, var, uns metadata)
adata = singlet.load("GSE136831")
print(adata.obs.columns)  # barcode, gsm_id, organism, total_counts
print(adata.uns["title"])  # Study title from GEO

# Load a single sample by column-range read
sample = singlet.load_sample("GSM3308814")

# Kraken2 microbiome matrix
k2 = singlet.read_kraken2("/path/to/GSE117795/")

# PyTorch DataLoader with normalization
from singlet.torch import DataLoader
loader = DataLoader("GSE136831", batch_size=512, normalize=True, device="cuda")
```

## Features

| Feature | Details |
|---------|---------|
| **Catalog** | Browse 3,309 datasets by organism, protocol, cell count |
| **Metadata** | Embedded obs/var/uns in .1pz — barcode, gsm_id, gene_name, study info |
| **Kraken2** | Per-dataset microbiome matrices (3,240 datasets) |
| **Compression** | singlepress .1pz format — ~13× vs raw CSC |
| **PyTorch** | `OnePZDataset` + `DataLoader` with log-normalization |
| **Sample access** | Column-range reads via `load_sample()` + sample_index |
| **Formats** | HDF5, Zarr, TileDB-SOMA, MTX, CSC sparse |
| **Local + Remote** | Local catalog (instant) or Zenodo download (free) |
| **Preprocessing** | Full FASTQ→.1pz pipeline (simpleaf + piscem) |

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/quickstart.md) | Getting started guide |
| [API Reference](docs/API.md) | Complete Python API |

## Part of Singlet AI

| Repository | Purpose |
|-----------|---------|
| [singlepress](https://github.com/Singlet-AI/singlepress) | Sparse matrix compression |
| [geo-reprocess](https://github.com/Singlet-AI/geo-reprocess) | HPC catalog pipeline |
| **singlet** | Python client library |
| [singlet-intelligence](https://github.com/Singlet-AI/singlet-intelligence) | ML models |
| [papers](https://github.com/Singlet-AI/papers) | Manuscripts |

## License

MIT
