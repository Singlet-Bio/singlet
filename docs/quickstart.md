# Quick Start

## Browse the Atlas

The bundled catalog works offline — no internet needed:

```python
import singlet

# One-line summary
print(singlet.summary())
# singlet atlas: 4,697 samples (2,072 SUCCESS) • 1,711 series • ...

# Search datasets
singlet.catalog("lung")

# Filter samples
singlet.samples(organism="Homo sapiens", tissue="brain", min_cells=1000)
```

## Load a Pre-Processed Sample

Every public GEO single-cell sample processed by the singlet pipeline is available:

```python
# Load by GEO accession (downloads from Zenodo)
adata = singlet.load("GSM5238385")
print(adata)
# AnnData object with n_obs × n_vars = 8234 × 33538

# Download only what you need: the exon matrix alone, or the splice junctions alone
spliced = singlet.load("GSM5238385", modalities=["spliced"])      # layers["spliced"], X is None
sj = singlet.load("GSM5238385", modalities=["splice_junctions"])  # obsm["splice_junctions"]
```

`modalities` takes `counts`, `spliced`, `unspliced`, `splice_junctions`, `psi`, `mt` and `vdj`
(see `docs/format-s1.md`, section 11); R's `load()` and `singlet_cells()` take the same names.

### Per-cell labels

Every per-sample study carries per-cell labels, cell calls and QC columns, and `obs` (R: `colData`) has them in front of the
GEO sample metadata:

```python
adata = singlet.load("GSE286650")
adata.obs[["cell_type", "cell_type_ontology_term_id", "cell_type_deepest", "sex_chrom", "ancestry_hancestro", "called_ed"]]
male_cm = singlet.cells("GSE286650", "cell_type == 'cardiac muscle cell' and sex_chrom == 'male'")
```

- `cell_type` / `cell_type_ontology_term_id`: the Cell Ontology label at level 2 or above (compartment, lineage), where the
  labeller is 0.84 / 0.75 accurate per study; `cell_type_deepest*` is the deeper call (0.43 / 0.32 at levels 3 / 4, a preview).
  `cell_type_level=6` puts the deepest call in `cell_type`.
- `sex_chrom` / `sex_score`: chromosomal sex, 98.1% agreement with GEO sample metadata.
- `ancestry_hancestro` / `ancestry_prob`: continental HANCESTRO category, human donors with coverage tables only.
- `called` is the default cell call, `called_ed` the EmptyDrops call (additional).
- `qc_doublet_score` is unvalidated and hidden unless `include_unvalidated=True`.

The full column list and what the accuracy numbers mean: `docs/cell_annotation.md`, section 8. With anndata 0.13 or newer,
`adata.layers` also lists `None`: that is anndata's name for `X` itself, not an extra layer.

## GPU-Accelerated Analysis

```python
from singlet import gpu
from singlet.gpu import preprocess as sgpp

# Full pipeline on GPU (10-50× faster than Scanpy for 100k+ cells)
sgpp.normalize_total(adata)
sgpp.log1p(adata)
sgpp.highly_variable_genes(adata, n_top_genes=2000)
gpu.reduce.pca(adata, n_comps=50)
gpu.pp.neighbors(adata, n_neighbors=15)
gpu.tools.leiden(adata, resolution=1.0)
gpu.tools.umap(adata)
```
