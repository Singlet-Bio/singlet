# singlet

**An open single-cell RNA-seq atlas — find, load, and analyze processed datasets with one line of Python.**

Singlet reprocesses public single-cell RNA-seq studies into a uniform,
analysis-ready atlas. You work with two simple things: GEO accession strings
(`GSE…` / `GSM…`) and `.singlet` files. Everything downloads as
[AnnData](https://anndata.readthedocs.io/), ready for scanpy or PyTorch.

- **Data: CC0** (public domain) — **Code: MIT**.
- No login. No API keys. No usage pricing. Public data is free to download.

## Install

```bash
pip install singlet
```

Optional extras:

```bash
pip install "singlet[torch]"   # PyTorch DataLoaders
pip install "singlet[gpu]"     # GPU-accelerated analysis (CUDA 12)
pip install "singlet[mcp]"     # Model Context Protocol server for AI assistants
pip install "singlet[all]"     # Everything
```

## Quick start

```python
import singlet

# Find datasets in plain English → list of accessions
accessions = singlet.find("human lung fibroblasts")

# Load an accession → AnnData (free download, cached locally)
adata = singlet.load("GSE149298")

# Load a local .singlet file
adata = singlet.load("path/to/data.singlet")

# Load and concatenate several datasets into ONE AnnData
adata = singlet.load(["GSE149298", "GSE264667", "extra.singlet"])

# Search and load in a single step
adata = singlet.find_load("human pancreas islet cells")
```

`load()` accepts a single accession, a single `.singlet` path, or a list/tuple
mixing both. Lists are concatenated into one AnnData with a `source` column in
`obs` recording where each cell came from.

## Natural-language search

```python
import singlet

singlet.find("exhausted T cells in melanoma")          # sample (GSM) accessions
singlet.find("mouse brain 10x", level="gse", limit=20) # series (GSE) accessions

# find + load in one call
adata = singlet.find_load("human kidney organoids")
```

Search runs against the hosted endpoint (`https://singlet.bio/api`; override with
`$SINGLET_API_BASE`).

## Browse the catalog (offline)

```python
import singlet

singlet.summary()                                   # atlas overview
singlet.samples(organism="Homo sapiens", status="SUCCESS")
singlet.tissues()                                   # tissue breakdown
singlet.cell_types()                                # cell-type breakdown
singlet.info("GSE264667")                           # dataset metadata
```

## PyTorch training

```python
from singlet.torch import SingletDataset, DataLoader

# Preferred: an array of .singlet files, concatenated into one training set
loader = DataLoader(
    ["a.singlet", "b.singlet", "c.singlet"],
    batch_size=512,
    device="cuda",
    normalize=True,
)

# Single dataset / accession / AnnData also work
ds = SingletDataset("GSE149298", normalize=True)
```

## Standard analysis (scanpy-compatible)

```python
import scanpy as sc

adata = singlet.load("GSE149298")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

## Links

- Homepage: https://singlet.bio
- Documentation: https://singlet.bio/docs
- Source: https://github.com/Singlet-Bio/singlet

## License

Code: MIT. Atlas data: CC0 (public domain).
</content>
