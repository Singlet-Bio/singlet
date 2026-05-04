# singlet

Python client for [SingletDB](https://singlet.bio) — the world's largest uniformly processed single-cell database.

## Overview

**singlet** provides access to 19,790+ uniformly-processed single-cell datasets across 24 species with ~10× compression via the [SinglePress](https://singlet-ai.github.io/singlepress/) format.

- **Browse & Download** — Free access to all datasets via Zenodo
- **PyTorch Integration** — Zero-copy sparse tensors with GPU support
- **Format Conversion** — Export to HDF5, Zarr, TileDB-SOMA, MTX
- **NMF Projection** — Server-side biological annotation
- **Cross-Atlas Queries** — Structured search by tissue, cell type, disease, organism

## Installation

```bash
pip install singlet

# With PyTorch support
pip install singlet[torch]
```

## Quick Start

```python
import singlet

# Browse datasets
df = singlet.catalog(search="lung")

# Load as AnnData
adata = singlet.load("GSE136831")

# PyTorch sparse tensor (GPU)
from singlet.torch import to_sparse_csr
tensor = to_sparse_csr("~/.singlet/data/GSE136831.1pz", device="cuda")
```

```{toctree}
:maxdepth: 2
:caption: Contents

install
quickstart
guide/index
api/index
```

## Related Projects

- [singlepress](https://singlet-ai.github.io/singlepress/) — Sparse matrix compression engine
- [geo-reprocess](https://singlet-ai.github.io/geo-reprocess/) — HPC pipeline that produces SingletDB data
- [singlet-intelligence](https://singlet-ai.github.io/singlet-intelligence/) — ML models and architecture
