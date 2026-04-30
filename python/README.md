# singlet

**Python client for the Singlet single-cell atlas.**

2,330 samples • 964 successful • 2.8M cells • 7 species • 1,153 GEO series • ~13× .1pz compression

## Install

```bash
pip install singlet-bio
```

## Quick Start

```python
import singlet

# Browse the atlas catalog (free, works offline)
singlet.summary()
# → 2,330 samples, 964 SUCCESS, 7 species, 2.83M cells

df = singlet.samples(organism="Homo sapiens", status="SUCCESS")
singlet.species()
singlet.top_series(n=10)

# Load a singlify output directory → AnnData
adata = singlet.load_dir("/path/to/quant/GSM3573650")
# → 75,420 cells × 38,606 genes
# obs: total_umis, total_genes, mt_pct, ribo_pct, intronic_pct,
#      doublet_score, is_doublet
# var: gene_id (Ensembl)

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
| **Catalog** | Browse 2,330 samples by organism, protocol, status, quality tier |
| **load_dir()** | Read singlify output directory → AnnData with QC + doublets |
| **read_1pz()** | Read .1pz sparse matrix → AnnData |
| **Compression** | singlepress .1pz format — ~13× vs raw CSC |
| **PyTorch** | `OnePZDataset` + `DataLoader` with log-normalization |
| **Offline** | Bundled catalog parquet — no network needed for browsing |

## Notebooks

7 executed Jupyter notebooks at [`notebooks/`](notebooks/):
- **quickstart** — Atlas catalog exploration
- **gene_counting** — Equivalence vs STARsolo (r=0.9995)
- **sex_calling** — XIST/SRY validation (100% agreement)
- **ambient_rna** — Ambient contamination profiling
- **doublet_detection** — UMI-based doublet scoring
- **corpus_analytics** — Atlas-wide QC distributions
- **01_load_and_explore** — Full scanpy pipeline on 75K cells

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
