# Changelog

## [1.2.0] — 2026-05-15

### Added
- `failure_categories()` — pipeline failure breakdown (5 categories: download_fail, align_low_map, cells_below_threshold, unknown, pipeline_crash)
- `tissues()` — 32 normalized tissue categories with 91% SUCCESS coverage
- `protocols()` — protocol distribution across 28 distinct protocols
- `quality_tiers()` — gold/silver/bronze classification with metrics
- MCP `singlet_failures` tool (10th tool total)
- Tests for `failure_categories()` and `tissues()` (131 passing)

### Changed
- Corpus expanded: 2,728 samples / 1,143 SUCCESS / 576 series / 3.3M cells / 16 species
- Tissue normalization: parses characteristics dict + source field + keyword matching
- MCP server now has 10 tools (was 9)
- quickstart notebook updated with all v1.2.0 API features
- README stats updated (2,716 samples, 121→32 normalized tissues, 10 MCP tools)
- pyproject.toml description refreshed

## [1.1.0] — 2026-04-30

### Added
- `load_dir()` — load full singlify output directory → AnnData with QC metrics, doublet scores, cell cycle, ancestry, sex call
- `summary()` — print atlas summary stats
- `samples()` — query individual samples (not just series)
- `species()` — list all species with sample counts
- `top_series()` — show highest-cell-count series
- 18 reproducibility notebooks with embedded matplotlib plots
- `notebooks/build_html.sh` — batch HTML generation for website hosting
- Google Colab integration via GitHub URL pattern
- GitHub Actions CI: Python 3.9/3.11/3.12 matrix + build verification

### Changed
- Catalog refreshed: 1,180 series / 2,398 samples / 998 SUCCESS / 2.94M cells
- README updated with current stats and `load_dir()` examples
- Tests expanded: 129 passing (up from 99)

## [1.0.0] — 2026-04-04

### Added
- `read_kraken2()` — load per-GSE microbiome matrices from kraken2.1pz
- `load_sample()` — load a single GSM sample via column-range reads
- `sample_index()` — access per-GSM column offsets for partial reads
- `set_catalog_dir()` — configure local catalog path (or SINGLET_CATALOG_DIR env var)
- `OnePZDataset` — PyTorch Dataset for .1pz with log-normalization support
- `DataLoader` now uses `OnePZDataset` with normalize parameter
- `docs/API.md` — complete API reference
- `CHANGELOG.md`

### Changed
- `_catalog.py` — rewritten for catalog_v1.parquet schema (gse_id, organism, n_cells, etc.)
- `_loader.py` — resolves GSE accessions to local .1pz files before Zenodo download
- `_loader.py` — downloads .1pz files (not .spz) for new datasets
- `datasets()` — new filters: organism, protocol, min_cells, has_kraken2
- `load()` — auto-detects local catalog, supports .1pz and .spz
- `docs/quickstart.md` — updated for new API
- `README.md` — updated with catalog v1.0 stats and new features
- `pyproject.toml` — version 0.2.0 → 1.0.0

### Deprecated
- `tissues()` function removed (not in catalog_v1 schema)
- `SpzDataset` — use `OnePZDataset` instead

## [0.2.0] — 2026-03

### Added
- .1pz format support (VOCSC + byte-split + zstd-3)
- Embedded obs/var/uns metadata in read_1pz
- write_1pz with include_obs/include_var/include_uns options
- Legacy .spz v1 and v2 format support
- PyTorch sparse CSR/COO tensors
- AnnData, Zarr, TileDB-SOMA conversions
- FASTQ→.1pz preprocessing pipeline
- Cross-atlas query and semantic search
