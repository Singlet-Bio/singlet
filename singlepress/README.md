# singlepress

[![Documentation](https://img.shields.io/badge/docs-singlet--ai.github.io-blue)](https://singlet-ai.github.io/singlepress/)

Compressed sparse matrix I/O for single-cell genomics — Python, R, and C++.

**`.1pz`** format — VOCSC + bit-plane bitmap + zstd/LZ4. Median 9.5× compression, 868 MB/s decode across 3,253 scRNA-seq datasets.

## Features

- **Median 9.5× compression** vs raw int32 CSC (range 2.4–12.1×, 3,253 datasets)
- **868 MB/s median decode** with OpenMP parallel chunk decompression
- **VOCSC encoding** — value-ordered column-sparse coding with adaptive gap width
- **Column-chunked CSC** — partial column and row reads, random access
- **CRC32 integrity** — per-chunk and whole-file checksums
- **AnnData-compatible metadata** — embedded obs, var, and uns DataFrames
- **Zero-copy PyTorch dataloaders** — decode to cuSPARSE CSR tensors for GPU training
- **Ecosystem interop** — AnnData, Seurat, SingleCellExperiment, HDF5, 10x .h5
- **Cross-language** — identical binary format across Python, R, and C++

## Installation

```bash
pip install singlepress                    # Core
pip install singlepress[torch]             # + PyTorch dataloaders
pip install singlepress[bio]               # + AnnData interop
pip install singlepress[all]               # Everything
```

## Quick Start

```python
import singlepress as sp

# Write/Read .1pz
mat = scipy.sparse.random(30000, 10000, format="csc")
sp.write_1pz("counts.1pz", mat, rownames=genes, colnames=cells,
              obs=cell_metadata, var=gene_metadata)
mat2 = sp.read_1pz("counts.1pz")

# PyTorch dataloader (zero-copy to GPU)
from singlepress.torch import OnePZDataset
dataset = OnePZDataset("counts.1pz", normalize=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=4)

# AnnData interop
from singlepress.interop import to_anndata, from_anndata
adata = to_anndata("counts.1pz")
from_anndata(adata, "output.1pz")

# HDF5 conversion
from singlepress.interop import from_10x_h5, to_h5ad
from_10x_h5("filtered_feature_bc_matrix.h5", "counts.1pz")
to_h5ad("counts.1pz", "counts.h5ad")
```

### R

```r
mat <- read_1pz("counts.1pz")           # dgCMatrix
obj <- read_1pz_seurat("counts.1pz")    # Seurat object
sce <- read_1pz_sce("counts.1pz")       # SingleCellExperiment
write_1pz(mat, "output.1pz")            # Write from R
```

## Format

`.1pz` is the sole format supported by singlepress. The legacy `.spz` format is no longer written or read.

See [docs/FORMAT_SPEC.md](docs/FORMAT_SPEC.md) for the complete binary format specification.

## Documentation

Full documentation at [singlet-ai.github.io/singlepress](https://singlet-ai.github.io/singlepress/).

| Guide | Description |
|-------|-------------|
| [Quick Start](https://singlet-ai.github.io/singlepress/quickstart.html) | Core read/write API |
| [Format Spec](docs/FORMAT_SPEC.md) | .1pz binary format |
| [PyTorch Dataloaders](https://singlet-ai.github.io/singlepress/pytorch_dataloader.html) | Zero-copy GPU training |
| [AnnData Interop](https://singlet-ai.github.io/singlepress/anndata_interop.html) | scanpy ecosystem |
| [R Integration](https://singlet-ai.github.io/singlepress/r_integration.html) | Seurat, SCE, dgCMatrix |
| [C++ API](https://singlet-ai.github.io/singlepress/cpp_api.html) | HPC/embedded usage |

## Part of Singlet AI

| Repository | Purpose |
|-----------|---------|
| [geo-reprocess](https://github.com/Singlet-AI/geo-reprocess) | HPC pipeline |
| [singlet](https://github.com/Singlet-AI/singlet) | Python client |
| **singlepress** | Sparse matrix compression |
| [singlet-intelligence](https://github.com/Singlet-AI/singlet-intelligence) | ML models |
| [singlet-strategy](https://github.com/Singlet-AI/singlet-strategy) | Strategic planning |
| [singletai-website](https://github.com/Singlet-AI/singletai-website) | Website |
| [papers](https://github.com/Singlet-AI/papers) | Manuscripts & reports |

## License

GPL-3.0-or-later
