# Author Metadata Extraction

## Overview

The metadata module extracts **all** author-deposited cell-level metadata for each processed GSM/GSE and aligns it to our quantified cell barcodes. This includes cell type labels, cluster assignments, donor IDs, treatment conditions, QC metrics, UMAP coordinates — everything authors provide.

Metadata is gathered from three tiers:

| Tier | Source | Coverage | Granularity |
|------|--------|----------|-------------|
| 1 | GEO SOFT characteristics | 100% | Sample-level |
| 2 | Supplementary files (h5ad, RDS, CSV/TSV, loom) | ~16% | Cell-level |
| 3 | NCBI E-utilities descriptions | 100% | Experiment-level |

## Quick start

```python
from scgeo.metadata import build_metadata

result = build_metadata("GSE281311", "GSM8618551")
print(result.status)     # "success" or "partial"
print(result.obs.head()) # Cell-level annotations aligned to our barcodes
print(result.uns)        # Experiment-level descriptions
```

## Architecture

### Tier 1 — SOFT characteristics

Every GEO submission includes structured sample-level metadata in the SOFT format. Common fields include `tissue`, `cell_type`, `treatment`, `genotype`, `age`, and `sex`. These are broadcast to all cells in a sample.

```python
from scgeo.metadata.soft import load_soft_metadata, parse_characteristics

soft = load_soft_metadata("GSE281311")
chars = parse_characteristics(soft["samples"]["GSM8618551"]["characteristics"])
# {'tissue': 'brain', 'cell_type': 'neuron', ...}
```

### Tier 2 — Supplementary file extraction

Authors deposit cell-level annotations in supplementary files. The module supports four formats:

**h5ad (AnnData)**

```python
from scgeo.metadata.extract import extract_metadata_from_h5ad
obs = extract_metadata_from_h5ad(Path("GSE295345_adata.h5ad"))
# Returns ALL .obs columns — cell types, clusters, QC, UMAP, etc.
```

**CSV / TSV / TXT**

```python
from scgeo.metadata.extract import extract_metadata_from_tabular
obs = extract_metadata_from_tabular(Path("GSE281311_metadata.tsv.gz"))
# Autodetects barcode column and separator
```

**RDS (Seurat / SingleCellExperiment)**

```python
from scgeo.metadata.extract_rds import extract_metadata_from_rds
obs = extract_metadata_from_rds(Path("GSE123456_seurat.rds"))
# Extracts meta.data from Seurat or colData from SCE objects
```

**Loom**

```python
from scgeo.metadata.extract import extract_metadata_from_loom
obs = extract_metadata_from_loom(Path("GSE123456.loom"))
# Reads all column attributes
```

### Tier 3 — NCBI descriptions

Natural language metadata (title, summary, organism, PubMed links) is fetched from the NCBI GDS database:

```python
from scgeo.metadata.description import fetch_geo_description
desc = fetch_geo_description("GSE281311")
# {'title': '...', 'summary': '...', 'organism': '...', 'pubmed_ids': [...]}
```

## Barcode matching

Author barcodes rarely match our processed barcodes exactly. The module tries three strategies:

1. **Direct match** — exact string equality
2. **Normalized** — strip prefixes (`SampleName_`) and suffixes (`-1`)
3. **Suffix-stripped** — remove only the gem-well suffix

Unmatched cells receive `NaN` for all annotation columns. Match rates depend on whether the author used the same quantification tool — typically 70–85% for tabular files, but can be near 0% for h5ad files from different pipelines (e.g., CellRanger vs simpleaf).

```python
from scgeo.metadata.barcodes import normalize_barcode, match_barcodes

normalize_barcode("Vehicle1_AAACCCAAGCATGAAT-1")
# 'AAACCCAAGCATGAAT'

mapping = match_barcodes(author_barcodes, our_barcodes)
# {author_bc: our_bc, ...}
```

## File classification

Before downloading, supplementary files are classified by format and prioritized:

```python
from scgeo.metadata import classify_supplementary_files

classified = classify_supplementary_files("GSE281311")
# {'h5ad': [...], 'rds': [...], 'tabular': [...], 'loom': [...], 'skip': [...]}
```

Files matching metadata-related patterns in their names (`meta`, `annot`, `cell_type`, etc.) are tried first.

## Batch processing

Process an entire GSE:

```python
from scgeo.metadata import build_metadata_gse

results = build_metadata_gse("GSE281311")
for gsm_id, result in results.items():
    print(f"{gsm_id}: {result.status}, {len(result.obs)} cells, "
          f"{len(result.tier2_columns)} Tier 2 columns")
```

Process multiple GSEs:

```python
from scgeo.metadata import build_metadata_batch

all_results = build_metadata_batch(["GSE281311", "GSE295345"])
```

## Output format

The `MetadataResult.obs` DataFrame is aligned to our processed cell barcodes (from `cells.parquet`) and ready to be used as AnnData `.obs`:

```python
import anndata as ad
import scipy.sparse

# Load counts
# ... (load SPZ matrix as sparse matrix) ...

adata = ad.AnnData(X=counts, obs=result.obs)
adata.uns.update(result.uns)
adata.write_h5ad("GSM8618551_annotated.h5ad")
```

## Dependencies

| Tier | Required | Optional |
|------|----------|----------|
| 1 | `pandas`, `pyarrow` | — |
| 2 (h5ad) | `anndata` or `h5py` | — |
| 2 (tabular) | `pandas` | — |
| 2 (RDS) | `rpy2`, R with `Seurat`/`SingleCellExperiment` | — |
| 2 (loom) | `loompy` | — |
| 3 | `urllib` (stdlib) | — |
