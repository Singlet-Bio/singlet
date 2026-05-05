# Catalog Building & Filtering

## Overview

The catalog system discovers single-cell RNA-seq series in NCBI GEO and enriches them with metadata from multiple sources.

## Discovery pipeline

The `build_catalog()` function runs a 5-stage pipeline:

1. **ESearch** — Find single-cell series UIDs in GEO
2. **ESummary** — Fetch series metadata (title, summary, organism)
3. **SOFT parsing** — Download and parse detailed sample metadata
4. **ENA/SRA queries** — Get FASTQ URLs, BioProject, and run information
5. **Protocol inference** — Detect chemistry (10x-v2/v3/v4, Drop-seq, etc.) from metadata hints

```python
import scgeo

catalog = scgeo.build_catalog(
    output="catalog.parquet",
    query="single cell RNA-seq",  # Custom search query
)
```

## Filtering

```python
filtered = scgeo.filter_catalog(
    catalog,
    organisms=["Homo sapiens", "Mus musculus"],
    min_samples=5,
    max_samples=500,
)
```

## Statistics

```python
stats = scgeo.get_catalog_stats(catalog)
```

## CLI

```bash
sc-geo catalog build --output catalog.parquet
sc-geo catalog discover --output discovery.json
sc-geo catalog filter --input catalog.parquet --organisms "Homo sapiens"
```
