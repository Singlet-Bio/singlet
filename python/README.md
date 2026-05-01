# singlet

**Python client for the Singlet single-cell atlas.**

2,708 samples • 1,138 successful • 3.3M cells • 9 species • 572 GEO series • 476 tissues • 8.7× .1pz compression

## Install

```bash
pip install singlet-bio
```

## Quick Start

```python
import singlet

# Browse the atlas catalog (free, works offline)
singlet.summary()
# → 2,708 samples, 1,138 SUCCESS, 6 species, 3.3M cells

df = singlet.samples(organism="Homo sapiens", status="SUCCESS")
singlet.samples(tissue="brain", status="SUCCESS")  # filter by tissue
singlet.species()
singlet.tissues()   # tissue/source breakdown
singlet.top_series(n=10)

# Load a singlify output directory → AnnData
adata = singlet.load_dir("/path/to/quant/GSM3573650")
# → 75,420 cells × 38,606 genes
# obs: total_umis, total_genes, mt_pct, ribo_pct, intronic_pct,
#      doublet_score, is_doublet, phase, s_score, g2m_score
# uns: ancestry, sex_call, summary, saturation_curve, singlify_dir

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
| **Catalog** | Browse 2,708 samples by organism, tissue, protocol, status, quality tier |
| **load_dir()** | Read singlify output directory → AnnData with QC, doublets, cell cycle, ancestry, sex, summary |
| **read_1pz()** | Read .1pz sparse matrix → AnnData |
| **Compression** | singlepress .1pz format — 8.7× vs h5ad |
| **PyTorch** | `OnePZDataset` + `DataLoader` with log-normalization |
| **Offline** | Bundled catalog parquet — no network needed for browsing |
| **MCP Server** | AI assistant access via `python -m singlet.mcp.server` |

## Notebooks

18 executed Jupyter notebooks at [`notebooks/`](notebooks/):
- **quickstart** — Atlas catalog exploration
- **gene_counting** — Equivalence vs STARsolo (r=0.9995)
- **sex_calling** — XIST/SRY validation (100% agreement)
- **ambient_rna** — Ambient contamination profiling
- **doublet_detection** — UMI-based doublet scoring
- **corpus_analytics** — Atlas-wide QC distributions
- **01_load_and_explore** — Full scanpy pipeline on 75K cells
- **cell_cycle** — Phase scoring (G1/S/G2M)
- **sample_qc_report** — Complete one-call QC report
- **saturation_curve** — Sequencing depth analysis
- **ancestry_calling** — Genetic ancestry inference
- **mt_variants** — Mitochondrial heteroplasmy
- **splicing** — Alternative splicing events
- **rna_velocity** — Spliced/unspliced for scVelo
- **pipeline_outputs** — Complete outputs reference
- **cell_calling** — EmptyDrops deviance testing
- **protocol_detection** — Corpus-wide protocol analysis
- **1pz_format** — .1pz format benchmarks (8.7× smaller than h5ad)

## Documentation

| Document | Description |
|----------|-------------|
| [Quick Start](docs/quickstart.md) | Getting started guide |
| [API Reference](docs/API.md) | Complete Python API |

## Part of Singlet

| Repository | Purpose |
|-----------|---------|
| [singlepress](https://github.com/Singlet-Bio/singlepress) | Sparse matrix compression |
| [geo-reprocess](https://github.com/Singlet-Bio/geo-reprocess) | HPC catalog pipeline |
| **singlet** | Python client library |
| [singlet-intelligence](https://github.com/Singlet-Bio/singlet-intelligence) | ML models |
| [papers](https://github.com/Singlet-Bio/papers) | Manuscripts |

## License

MIT
