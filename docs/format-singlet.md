# `.singlet` Bundle Format Specification

Version: **1.0** | Magic: ZIP64 archive with extension `.singlet`

---

## Overview

A `.singlet` file is a **ZIP64 archive** containing all processed GSMs for one
GEO Series (GSE). It is the primary distribution unit for the Singlet atlas.
One file per study. No Cloudflare or remote storage required for local use.

```
GSE122083.singlet          ← ZIP64 archive
├── manifest.json          schema, GSM list, checksums, versions
├── study_meta.json        series metadata + per-GSM obs metadata
├── feature_vocab.json     gene ↔ Ensembl map (shared across all GSMs)
└── samples/
    ├── GSM3454528/
    │   ├── exon_counts.1pz      STORED (zstd already inside)
    │   ├── intron_counts.1pz    STORED
    │   ├── sj_counts.1pz        STORED (if present)
    │   ├── splice_psi.1pz       STORED (if present)
    │   ├── vdj_gene_usage.1pz   STORED (if present)
    │   ├── mt_heteroplasmy.1pz  STORED (if present)
    │   ├── summary.json         DEFLATED
    │   ├── pileup_stats.json    DEFLATED
    │   ├── provenance.json      DEFLATED
    │   └── cell_calls.tsv       DEFLATED  ← called-cell barcode list (is_cell=True rows)
    └── GSM3454529/
        └── ...
```

---

## ZIP Compression Policy

| File type | ZIP method | Reason |
|-----------|-----------|--------|
| `*.1pz`          | `STORED` (method 0)   | Already contains zstd-compressed frames; double-compressing wastes CPU with no gain |
| `*.json`         | `DEFLATED` (method 8) | Small, highly compressible text; 3–10× size reduction typical |
| `cell_calls.tsv` | `DEFLATED` (method 8) | Small TSV; required by reader to filter called cells from the full whitelist in .1pz |

ZIP64 extensions are always enabled (`allowZip64=True`) to support bundles
exceeding 4 GB.

---

## `manifest.json` — Schema

```json
{
  "schema_version": "1.0",
  "gse_id": "GSE122083",
  "gsm_ids": ["GSM3454528", "GSM3454529"],
  "reference_build": "GRCh38-2024-A",
  "n_gsms": 2,
  "created_at": "2026-06-18T00:00:00+00:00",
  "singlet_version": "0.3.0",
  "included_files": {
    "GSM3454528": ["exon_counts.1pz", "intron_counts.1pz", "summary.json",
                   "pileup_stats.json", "provenance.json"],
    "GSM3454529": ["..."]
  },
  "checksums": {
    "feature_vocab.json": "<sha256>",
    "study_meta.json": "<sha256>",
    "samples/GSM3454528/exon_counts.1pz": "<sha256>",
    "...": "..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | str | Always `"1.0"` for this spec |
| `gse_id` | str | GEO Series accession |
| `gsm_ids` | list[str] | All GSMs included (in packing order) |
| `reference_build` | str | Genome/annotation build (e.g. `"GRCh38-2024-A"`) |
| `n_gsms` | int | `len(gsm_ids)` |
| `created_at` | str | ISO 8601 UTC timestamp |
| `singlet_version` | str | `singlet` package version used to create the bundle |
| `included_files` | dict | GSM → list of filenames stored in `samples/<GSM>/` |
| `checksums` | dict | Arc path → SHA-256 hex digest for every member file |

---

## `study_meta.json` — Schema

```json
{
  "schema_version": "1.0",
  "gse_id": "GSE122083",
  "series_title": "Prediction of bacterial infection ...",
  "series_summary": "...",
  "series_overall_design": "...",
  "publications": [],
  "gsm_meta": {
    "GSM3454528": {
      "gsm_id": "GSM3454528",
      "organism": "Homo sapiens",
      "protocol": "10xv3",
      "protocol_name": "10xv3",
      "sample_source": "PBMCs",
      "sample_characteristics": "infection: control ;; time: 0 hr ;; cell type: PBMCs",
      "qc_flag": null,
      "reference_build": "GRCh38-2024-A",
      "n_cells": 3481,
      "median_umi_per_cell": 2847,
      "median_genes_per_cell": 1234,
      "mapping_rate": 0.8914
    }
  }
}
```

### `gsm_meta[<GSM>]` fields

| Field | Source | Description |
|-------|--------|-------------|
| `gsm_id` | catalog | GEO Sample accession |
| `organism` | catalog `organism` | e.g. `"Homo sapiens"` |
| `protocol` | catalog `protocol_inferred` | e.g. `"10xv3"` |
| `protocol_name` | `summary.json` | Pipeline-detected protocol |
| `sample_source` | catalog `sample_source` | Tissue/cell type label |
| `sample_characteristics` | catalog `sample_characteristics` | GEO characteristics string |
| `qc_flag` | catalog `qc_flag` | Pipeline QC flag (null = pass) |
| `reference_build` | `summary.json` | Genome + annotation build |
| `n_cells` | `summary.json` `n_cells_called` | Number of called cells |
| `median_umi_per_cell` | `summary.json` | Median UMIs per cell |
| `median_genes_per_cell` | `summary.json` | Median genes per cell |
| `mapping_rate` | `summary.json` | Fraction of reads uniquely mapped |

`publications` is a placeholder list; it will be populated by a future enrichment
step using PubMed cross-references.

---

## `feature_vocab.json` — Schema

```json
{
  "schema_version": "1.0",
  "reference_build": "GRCh38-2024-A",
  "n_genes": 38606,
  "genes": [
    {"gene_id": "ENSG00000000003", "gene_name": "TSPAN6"},
    {"gene_id": "ENSG00000000005", "gene_name": "TNMD"},
    "..."
  ]
}
```

- Derived from `gene_expression.tsv` in the first included GSM's output.
- All GSMs in the bundle share the same gene axis (same `reference_build`).
- `gene_id` = Ensembl ID. `gene_name` = HGNC symbol (or equivalent).
- Gene order matches the rows of the count matrices returned by `to_anndata()`.

---

## Per-GSM `.1pz` Files

Each GSM directory stores the raw pipeline `.1pz` outputs verbatim (STORED in
ZIP). The pipeline v1 format uses `TP1Z` magic; the v2 format uses `1PZ02`.
Both are supported by `singlet._io._read_pz_native()` and `singlet.pz_v2`.

| File | Content |
|------|---------|
| `exon_counts.1pz` | Per-exon-interval UMI counts (features × all barcodes) |
| `intron_counts.1pz` | Per-intron-interval UMI counts |
| `sj_counts.1pz` | Per-splice-junction UMI counts |
| `splice_psi.1pz` | Per-junction percent-spliced-in |
| `vdj_gene_usage.1pz` | VDJ gene segment usage |
| `mt_heteroplasmy.1pz` | Mitochondrial variant allele frequency |

Gene-level counts are derived on-the-fly by `SingletBundle.to_anndata()` by
summing exon + intron intervals per Ensembl gene ID.

---

## Python API

### Packing

```python
from singlet.bundle import pack_gse

bundle_path = pack_gse(
    gse_id="GSE122083",
    results_dir="/path/to/results",
    catalog_path="/path/to/processing_catalog.parquet",
    out_path="/path/to/GSE122083.singlet",
)
```

### CLI

```bash
python -m singlet.bundle pack \
    --gse GSE122083 \
    --results /path/to/results \
    --catalog /path/to/processing_catalog.parquet \
    --out GSE122083.singlet
```

### Reading

```python
from singlet.bundle import SingletBundle

bundle = SingletBundle.open("GSE122083.singlet")
print(bundle.manifest)
print(bundle.gsm_ids)

# Load full AnnData
adata = bundle.to_anndata()
# adata.shape == (n_total_cells, n_genes)
# adata.obs has: gsm_id, organism, protocol, sample_source,
#               sample_characteristics, qc_flag, reference_build
# adata.var has: gene_name; index = Ensembl gene_id
# adata.uns has: study_meta, manifest

# Export
bundle.to_h5ad("GSE122083.h5ad")
bundle.to_parquet("GSE122083_obs.parquet")  # metadata only
```

---

## Size Guidance for Large GSEs

| Scenario | Typical bundle size | Notes |
|----------|--------------------|-|
| 2–5 GSMs, ~5K cells each | 200 MB – 1 GB | Fits in RAM for `to_anndata()` |
| 10–50 GSMs | 1–10 GB | Use `to_h5ad()` for streaming write |
| >100 GSMs | 10–100+ GB | Consider splitting by tissue/batch; ZIP64 handles size |

**Recommendation for large GSEs:** Call `to_h5ad()` directly rather than
`to_anndata()` to avoid accumulating all sparse matrices in RAM. For
billion-cell-scale studies, the bundle format is not the primary access layer
— use TileDB-SOMA or Zarr stores instead, built from individual GSM outputs.

The packager streams files into ZIP without buffering the whole archive,
so pack time scales linearly with total data size regardless of GSE size.

---

## Versioning

The bundle schema version is independent of the `singlet` package version.
Both are recorded in `manifest.json`. Forward compatibility is maintained:
readers should ignore unknown keys in JSON members.

| Schema version | Changes |
|----------------|---------|
| `1.0` | Initial specification |
