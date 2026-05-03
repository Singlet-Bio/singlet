# Quick Start

## Install

```bash
pip install singlet-bio
```

## Browse the Atlas

The atlas catalog is bundled with the package — works offline, no API key needed:

```python
import singlet

# One-line atlas summary
singlet.summary()
# → 'singlet atlas: 4,506 samples (1,975 SUCCESS) • 1,679 series • 8 species • 28 protocols • 37 tissues • 40 cell types • 5.4M cells'

# Browse all samples
df = singlet.samples(status="SUCCESS")
print(f"{len(df)} successful samples")

# Filter by organism, tissue, cell type, protocol
brain = singlet.samples(tissue="brain", organism="Homo sapiens")
pbmc = singlet.samples(cell_type="PBMC")
gold = singlet.samples(quality_tier="gold")

# Search by text (title, source, GSM/GSE)
lung_cancer = singlet.samples(search="lung cancer")

# Explore the atlas
singlet.species()        # → ['Drosophila melanogaster', 'Homo sapiens', 'Mus musculus', ...]
singlet.tissues()        # → DataFrame of 37 normalized tissue categories
singlet.cell_types()     # → DataFrame of 235 cell type categories
singlet.protocols()      # → DataFrame of 28 protocols (SUCCESS only)
singlet.quality_tiers()  # → gold/silver/bronze breakdown
singlet.top_series(n=5)  # → top 5 series by cell count

# Series-level catalog
cat = singlet.catalog(search="brain")
```

## Load Data

```python
# Load a singlify output directory → AnnData
adata = singlet.load_dir("/path/to/quant/GSM3573650")
print(adata)
# → 75,420 cells × 38,606 genes
# obs: total_umis, total_genes, mt_pct, doublet_score, is_doublet, phase, ...
# uns: ancestry, sex_call, summary, saturation_curve

# Load a .1pz file
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

## PyTorch Integration

```python
from singlet.torch import OnePZDataset, DataLoader

# Load as PyTorch dataset
dataset = OnePZDataset("counts.1pz", normalize=True)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

for batch in loader:
    # batch is (batch_size, n_genes) log-normalized tensor
    pass
```

## Update the Catalog

```python
# Download latest catalog from GitHub (when new samples are processed)
singlet.refresh()
```

## MCP Server (for AI Assistants)

```bash
# Expose atlas data to Claude, Cursor, or VS Code Copilot
python -m singlet.mcp.server
```

11 tools available: stats, search, browse, QC, load, protocols, quality, tissues, failures, cell_types, species.
