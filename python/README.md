# singlet

**Python client for the Singlet single-cell atlas.**

4,697 samples • 2,072 successful • 5.7M cells • 8 species • 37 tissues • 40 cell types • 28 protocols • 8.7× .1pz compression

## Install

```bash
pip install singlet
```

Optional extras:
```bash
pip install singlet[torch]   # PyTorch DataLoaders
pip install singlet[gpu]     # GPU-accelerated analysis
pip install singlet[all]     # Everything
```

## Quick Start

```python
import singlet

# Browse the atlas catalog
singlet.summary()
df = singlet.samples(organism="Homo sapiens", status="SUCCESS")
singlet.tissues()
singlet.cell_types()

# Load a pipeline output directory → AnnData
adata = singlet.load_dir("/path/to/quant/GSM3573650")

# Load a .1pz file directly
adata = singlet.read_1pz("/path/to/gene_counts.1pz")

# Standard scanpy workflow
import scanpy as sc
adata = adata[~adata.obs['is_doublet'].astype(bool)]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

## Features

| Feature | Details |
|---------|---------|
| **Catalog** | Browse 4,697+ samples by organism, tissue, cell type, protocol, status, quality tier |
| **tissues()** | 37 normalized tissue categories |
| **cell_types()** | 40 normalized cell type categories |
| **protocols()** | 28 distinct protocols (10xv3, Drop-seq, CEL-Seq2, sci-RNA-seq3, etc.) |
| **load_dir()** | Read pipeline output directory → AnnData with QC, doublets, cell cycle |
| **read_1pz()** | Read .1pz sparse matrix → AnnData |
| **Compression** | .1pz format — 8.7× vs h5ad |
| **PyTorch** | `OnePZDataset` + `DataLoader` with log-normalization |
| **GPU** | GPU-accelerated PCA, NMF, neighbors, leiden, UMAP |
| **MCP Server** | AI assistant tools via `python -m singlet.mcp.server` |

## PyTorch Integration

```python
from singlet.torch import OnePZDataset, DataLoader

dataset = OnePZDataset("sample.1pz", normalize=True)
loader = DataLoader("sample.1pz", batch_size=512, device="cuda")
```

## License

MIT
