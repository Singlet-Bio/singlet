# Catalog v1.0 Specification

> Per-GSE directory layout, metadata schemas, and index files for the Singlet single-cell RNA-seq catalog.

---

## 1. Overview

The catalog organizes reprocessed GEO single-cell datasets as **one directory per GSE accession**. Each directory contains a merged sparse-matrix file (`.1pz`), sidecar metadata in Parquet format, and provenance records.

**Catalog v1.0 totals**: 3,309 GSE datasets (3,377 with multi-species splits), 21,006 samples (GSMs), 354M cells, 337B non-zeros.

Base path: `pipeline/quant/`

---

## 2. Per-GSE Directory Layout

### 2.1 Single-Species GSE

```
GSE117795/
├── counts.1pz                  # Merged expression matrix (genes × cells)
├── metadata.parquet            # Per-cell annotations (obs)
├── feature_metadata.parquet    # Per-gene annotations (var)
├── study_metadata.json         # Study-level metadata (uns)
├── provenance.json             # Merge provenance record
├── kraken2.1pz                 # Microbiome matrix (taxa × cells) [optional]
├── kraken2_features.parquet    # Taxon metadata [optional]
├── GSM3308814/                 # Per-sample source directories
│   ├── counts.1pz              #   Original per-GSM matrix
│   ├── cell_metadata.parquet
│   ├── feature_metadata.parquet
│   └── kraken2_cell_taxa.parquet
├── GSM3308815/
│   └── ...
└── ...
```

### 2.2 Multi-Species GSE

When a GSE contains samples from multiple organisms, merged data is split into species subdirectories:

```
GSE100384/
├── Homo_sapiens/
│   ├── counts.1pz
│   ├── metadata.parquet
│   ├── feature_metadata.parquet
│   ├── study_metadata.json
│   └── provenance.json
├── Mus_musculus/
│   ├── counts.1pz
│   ├── metadata.parquet
│   ├── feature_metadata.parquet
│   ├── study_metadata.json
│   └── provenance.json
├── GSM2679965/
│   └── ...
└── ...
```

Species directory names use underscored binomial nomenclature (e.g., `Homo_sapiens`, `Mus_musculus`, `Macaca_mulatta`).

---

## 3. File Schemas

### 3.1 `counts.1pz`

Singlepress v3 format (see `singlepress/docs/FORMAT_SPEC.md`).

| Dimension | Meaning | Typical Size |
|-----------|---------|-------------|
| Rows | Genes / features | ~115K–172K |
| Columns | Cells (barcodes) | Varies by GSE |

Matrix values are UMI counts (unsigned integers). Value type auto-selected: uint8, uint16, or uint32.

Embedded metadata:
- **obs** (DataFrame): `barcode`, `gsm_id`, `organism`, `total_counts`
- **var** (DataFrame): `gene_name`, `reference`
- **uns** (KV pairs): `gse_id`, `title`, `organism`, `protocol`, `pubmed_ids`, etc.
- **rownames**: Gene names
- **colnames**: Not stored (use obs `barcode` column)

Cell ordering: GSMs appear in `source_gsms` order from `provenance.json`. Within each GSM, cells appear in `cell_metadata.parquet` order.

### 3.2 `metadata.parquet`

Per-cell (obs) annotations. One row per cell, same order as `counts.1pz` columns.

| Column | Type | Description |
|--------|------|-------------|
| `barcode` | string | Cell barcode sequence |
| `gsm_id` | string | Source GEO sample accession |
| `organism` | string | Species name |
| `total_counts` | int64 | Total UMI count for this cell |

### 3.3 `feature_metadata.parquet`

Per-gene (var) annotations. One row per gene, same order as `counts.1pz` rows.

| Column | Type | Description |
|--------|------|-------------|
| `gene_name` | string | Gene symbol or ID |
| `reference` | string | Reference genome identifier |

> **Note**: Some legacy GSMs use `gene_id` instead of `gene_name`. The merge pipeline normalizes to `gene_name` where possible.

### 3.4 `study_metadata.json`

Study-level metadata derived from GEO and the processing pipeline:

```json
{
    "gse_id": "GSE117795",
    "title": "Reprogram-Seq: ...",
    "summary": "...",
    "organism": ["Mus musculus"],
    "n_samples": 6,
    "n_cells": 571129,
    "n_genes": 171540,
    "gsm_ids": ["GSM3308814", "GSM3308815", ...],
    "protocol": "10xv2",
    "reference": "USA_171540_mouse",
    "pubmed_ids": "[\"31216470\"]",
    "submission_date": "",
    "license": "public_domain",
    "catalog_version": "1.0"
}
```

### 3.5 `provenance.json`

Merge provenance record — documents how per-GSM files were combined:

```json
{
    "created_at": "2026-04-03T21:28:46.205956+00:00",
    "merge_method": "hstack",
    "source_gsms": {
        "GSM3308814": {
            "pipeline_version": "2026-02-12_v3_pipelined_kraken2",
            "n_cells": 115410,
            "n_features": 171540,
            "organism": "Mus musculus",
            "protocol": "10xv2",
            "qc_status": "qc_pass"
        },
        ...
    }
}
```

### 3.6 `kraken2.1pz` (optional)

Microbiome count matrix in singlepress v3 format. Present for ~3,240 of 3,309 GSEs.

| Dimension | Meaning |
|-----------|---------|
| Rows | NCBI taxonomy IDs |
| Columns | Cells (same order as `counts.1pz`) |

Values are UMI counts assigned to each taxon per cell.

Embedded metadata:
- **uns**: `gse_id`, `type` ("kraken2_cell_taxa"), `n_taxa`, `n_cells`
- **rownames**: Taxon IDs as strings

### 3.7 `kraken2_features.parquet` (optional)

| Column | Type | Description |
|--------|------|-------------|
| `taxon_id` | int64 | NCBI taxonomy ID |
| `name` | string | Taxon ID as string (taxonomy name lookup deferred) |

---

## 4. Catalog Index Files

Located at `catalog/` (sibling to `pipeline/`).

### 4.1 `catalog_v1.parquet`

One row per GSE dataset (3,309 rows). Primary discovery index.

| Column | Type | Description |
|--------|------|-------------|
| `gse_id` | string | GEO series accession |
| `organism` | string | Species name(s), `\|`-separated if multi-species |
| `n_samples` | int64 | Number of GSM samples |
| `n_cells` | int64 | Total cell count |
| `n_genes` | int64 | Number of genes/features |
| `reference` | string | Reference genome ID(s) |
| `protocol` | string | Library protocol (10xv2, 10xv3, etc.) |
| `has_kraken2` | bool | Whether kraken2.1pz exists |
| `has_author_meta` | bool | Whether author-provided metadata exists |
| `license` | string | Data license |
| `path` | string | Relative path to GSE directory |
| `catalog_version` | string | Catalog version ("1.0") |

### 4.2 `sample_index.parquet`

One row per GSM sample (21,006 rows). Enables column-range access:

| Column | Type | Description |
|--------|------|-------------|
| `gsm_id` | string | GEO sample accession |
| `gse_id` | string | Parent GSE accession |
| `organism` | string | Species name |
| `n_cells` | int64 | Number of cells in this sample |
| `pipeline_version` | string | Processing pipeline version |
| `species_subdir` | string | Species subdirectory name (empty if single-species) |
| `col_offset` | int64 | Starting column index in `counts.1pz` |
| `col_count` | int64 | Number of columns for this sample |

Use `col_offset` and `col_count` with `singlepress.read_1pz_columns()` for sample-level random access.

---

## 5. Data Access Patterns

### 5.1 Load Entire GSE

```python
import singlepress as sp

mat = sp.read_1pz("pipeline/quant/GSE117795/counts.1pz")
# mat.shape = (171540, 571129)
# mat.obs → DataFrame with barcode, gsm_id, organism, total_counts
# mat.var → DataFrame with gene_name, reference
# mat.uns → dict with gse_id, title, etc.
```

### 5.2 Load Single Sample

```python
import pyarrow.parquet as pq

# Find sample offset
idx = pq.read_table("catalog/sample_index.parquet").to_pandas()
gsm = idx[idx.gsm_id == "GSM3308814"].iloc[0]

mat = sp.read_1pz_columns(
    f"pipeline/quant/{gsm.gse_id}/counts.1pz",
    gsm.col_offset,
    gsm.col_offset + gsm.col_count,
)
```

### 5.3 Discover Datasets

```python
catalog = pq.read_table("catalog/catalog_v1.parquet").to_pandas()

# All human datasets with >100K cells
large_human = catalog[
    (catalog.organism.str.contains("Homo sapiens")) &
    (catalog.n_cells > 100_000)
]
```

---

## 6. Naming Conventions

| Pattern | Example | Description |
|---------|---------|-------------|
| `GSE{digits}` | `GSE117795` | GEO series accession |
| `GSM{digits}` | `GSM3308814` | GEO sample accession |
| `{Genus}_{species}` | `Homo_sapiens` | Species subdirectory |
| `USA_{N}_{species}` | `USA_171540_mouse` | Reference genome identifier |
| `counts.1pz` | — | Merged expression matrix |
| `kraken2.1pz` | — | Merged microbiome matrix |

---

## 7. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-04 | Initial release: 3,309 GSEs, 354M cells |
