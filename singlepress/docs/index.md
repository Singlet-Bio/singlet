# singlepress

Compressed sparse matrix I/O for single-cell genomics.

## Overview

**singlepress** provides the `.1pz` format — a column-chunked, VOCSC-encoded sparse matrix format achieving **13× compression** with **4000+ MB/s decode speed**.

### Key Features

- **Column-chunked CSC storage** — independent chunks enable partial reads and parallel decode
- **VOCSC encoding** — value-ordered column-sparse coding with adaptive gap widths
- **Byte-split + zstd-3 compression** — exploits low-entropy byte planes for extreme ratios
- **CRC32 integrity** — per-chunk and whole-file checksums
- **AnnData-compatible metadata** — embedded obs, var, uns (cell/gene/study metadata)
- **Zero-copy PyTorch dataloaders** — decode directly to cuSPARSE-compatible CSR tensors
- **Cross-language** — Python, R, and C++ APIs, all reading/writing the same binary format
- **Ecosystem interop** — AnnData, Seurat, SingleCellExperiment, HDF5, 10x .h5

## Installation

```bash
pip install singlepress
```

Requires a C++17 compiler and libzstd.

## Quick Start

```python
import singlepress as sp
from scipy.sparse import random

# Write
mat = random(30000, 10000, density=0.05, format="csc")
sp.write_1pz("counts.1pz", mat,
              rownames=[f"gene_{i}" for i in range(30000)],
              colnames=[f"cell_{i}" for i in range(10000)])

# Read
mat2 = sp.read_1pz("counts.1pz")
print(mat2.rownames[:5], mat2.colsums[:5])

# PyTorch dataloader
from singlepress.torch import OnePZDataset
dataset = OnePZDataset("counts.1pz", normalize=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=4)
```

```{toctree}
:maxdepth: 2
:caption: User Guide

install
quickstart
onepz_class
dataset_operations
format_spec
```

```{toctree}
:maxdepth: 2
:caption: Ecosystem

pytorch_dataloader
anndata_interop
r_integration
cross_language
```

```{toctree}
:maxdepth: 2
:caption: Reference

api
cpp_api
cli_reference
```
