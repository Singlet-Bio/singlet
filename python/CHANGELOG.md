# Changelog

## [1.3.2] — Unreleased

### Added
- `singlet_species` MCP tool (11 tools total)
- MCP smoke test expanded to cover all 11 tools
- Blog post #39: "4 Million Cells: The Singlet Atlas Passes a Major Milestone"

### Changed
- Corpus: 4,345 samples / 1,899 SUCCESS / 828 series / 5.3M cells / 8 species / 29 protocols
- Parquets rebuilt with normalized tissue/cell_type categories (fixed duplicates like `bone_marrow` vs `bone marrow`)
- MCP README updated with all 11 tools

### Fixed
- Tissue/cell_type category duplicates in parquet (underscore vs human-readable naming)

## [1.3.1] — 2026-05-01

### Added
- `failure_categories()` now uses ground-truth data from pipeline (8 real categories vs 4 inferred)
- `failure_category` column added to sample_index.parquet
- `top_series()` now includes tissue column
- `scripts/rebuild_parquets.py` — automated Supabase → parquet rebuild script
- Blog post #38: "Cell Type Coverage Doubles to 50%"
- Prepared `publish_4m_blog.py` for automated 4M milestone publishing

### Changed
- `species()` now uses sample_index with proper deduplication (8 species, consistent with `summary()`)
- `summary()` species count uses deduplicated species (was counting combo organisms separately)
- Loader priority: bundled parquet > cache > download (fixes stale cache issue)
- `refresh()` loads data directly into cache vars (bypasses bundled-first after download)
- Corpus: 3,196 samples / 1,368 SUCCESS / 658 series / 3.97M cells / 8 species / 29 protocols
- Cell type coverage: 57% (40 categories, up from 24%)
- Tissue coverage: 73% (36 categories)
- GEO title coverage: 99.8%

### Fixed
- Stale cache served before bundled parquet (loader reordered)
- `refresh()` ineffective after bundled-first fix (now bypasses bundled)
- Tests: `organisms` → `organism`, `total_cells` → `n_cells` column name updates

## [1.3.0] — 2026-05-01

### Added
- `cell_types()` — 40 normalized cell type categories (PBMC, T cells, stem cells, K562, etc.) with 57% SUCCESS coverage
- `samples(cell_type=...)` — filter samples by cell type annotation
- MCP `singlet_cell_types` tool (11th tool total)
- Blog post #37: "3,000 Samples Processed" milestone

### Changed
- Corpus expanded: 3,128 samples / 1,330 SUCCESS / 648 series / 3.67M cells / 16 species
- QC enrichment: median_genes coverage 86% → 99.5% (from quant summary.json backfill)
- Tissue normalization: 36 categories, 71% SUCCESS coverage
- MCP server now has 11 tools (was 10)
- ETL sync script rewritten: batch metadata index, row-level fallback, proper error handling
- `summary()` now includes cell type count in one-liner
- README and pyproject.toml refreshed for 3,100+ stats

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
