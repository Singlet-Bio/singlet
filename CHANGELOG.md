# Changelog

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
