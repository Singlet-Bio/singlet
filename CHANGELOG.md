# Changelog

All notable changes to the singlet project.

## [2.0.0] — 2026-05-05

### Breaking Changes
- **Unified package**: `singlet-bio`, `singlepress`, `singlet-gpu` merged into single `singlet` package
- **Install**: `pip install singlet` (replaces `pip install singlet-bio`)
- **R package**: renamed from `singlify` to `singlet`
- **C++ namespaces**: `singlet::pz`, `singlet::fq`, `singlet::pileup`, `singlet::gpu`

### Features
- **Monorepo consolidation**: Single repo ships Python, R, and C++ library
- **Unified C++ library** (`libsinglet`): Header-only with CMake INTERFACE targets
- **Optional extras**: `pip install singlet[torch]`, `singlet[gpu]`, `singlet[all]`
- **STAR integration**: STAR aligner built as object library within the unified CMake project
- **PyTorch module**: `from singlet.torch import OnePZDataset, DataLoader`
- **GPU module**: `from singlet.gpu import ...` (requires cupy)
- **CMake find_package**: `find_package(Singlet COMPONENTS pz fq pileup)` with version file
- **R GPU support**: `singlet::has_gpu()`, `gpu_pca()`, `gpu_neighbors()`, `gpu_leiden()`
- **C++ test suite**: 100 unit tests covering all pileup modules (codec, cell calling, ATAC, ADT, species, nonhost, export, spatial, protocol detection, UMI dedup, bloom filter, velocity, saturation, read stats, provenance, minimizer index, cascade stats, pz writer, ancestry, ASE, MTX writer)
- **IO schema v2 support**: Loader auto-detects singlify v2 subdirectory layout (`donor/snp_ad.1pz`)
- **MCP server tests**: 25 unit tests covering all parquet-backed tools + call_tool router
- **Python lint**: ruff check + ruff format enforced (0 errors, CI job added)
- **Type annotations**: 75% of functions annotated with return types (236/312)
- **Makefile**: `make test`, `make lint`, `make build`, `make pipeline`, `make clean`
- **Python API reference**: `docs/api/python.md` covers all public modules
- **Annotation tests**: 15 unit tests for `_annotate.py` (gene_programs, project, annotate)
- **Security**: `SECURITY.md` vulnerability reporting policy + pip-audit CI job
- **Python tests**: 584 total (up from 205 at cycle 184, 98% core coverage)
- **Type checking**: pyright with 0 errors (basic mode, pyrightconfig.json)
- **Format conversion tests**: MTX round-trip, to_csc, from_mtx edge cases
- **Preprocessing tests**: protocol detection, FASTQ download, quantify, QC, species, export

### Architecture
- `include/singlet/pz/` — .1pz VOCSC codec (13x compression, 4000+ MB/s decode)
- `include/singlet/fq/` — .1fq 2-bit packed FASTQ codec
- `include/singlet/pileup/` — streaming BAM pileup engine (70+ modules)
- `include/singlet/gpu/` — CUDA analysis kernels
- `include/singlet/star/` — STAR aligner API (singlet-lite fork, 48% faster)

### Removed
- Standalone `singlepress` repository (archived)
- Standalone `singlet-gpu` repository (archived)
- Standalone `singlet-bio` repository (archived)
- Agent configs moved to private `singlet-agents` repo

## [1.0.0] — 2026-05-04

### Features
- **Bundled catalog**: Package ships with `catalog_v1.parquet` (1,175 series) and `sample_index.parquet` (2,378 samples). No downloads needed to browse the atlas.
- **Text search**: `singlet.samples(search="lung")` — full-text search across GEO titles, organisms, protocols.
- **Quality tiers**: `singlet.samples(quality_tier="gold")` — filter by mapping rate and cell count.
- **load_dir() v3**: Reads 10 singlify output files into a single AnnData with cell cycle phases, ancestry, sex call, pipeline summary, and saturation curves in `obs`/`uns`.
- **17 notebooks**: Complete reproducibility collection covering QC, genomic features, and validation.
- **MCP server**: `python -m singlet.mcp.server` exposes atlas data to AI assistants (requires Python 3.10+).

### Catalog API
- `singlet.summary()` — one-line atlas overview
- `singlet.catalog(search)` — browse series with text filter
- `singlet.samples(search, organism, status, min_cells, quality_tier, gse_id)` — rich sample queries
- `singlet.sample_index()` — full sample DataFrame with titles and QC
- `singlet.species()` — list of 7 species in the atlas
- `singlet.top_series(n, organism)` — largest series by cell count
- `singlet.datasets(organism, min_cells)` — filter series catalog
- `singlet.info(accession)` — series metadata dict

### Data Loading
- `singlet.load_dir(path)` — singlify output → AnnData (primary interface)
- `singlet.load(source)` — load local .1pz/.spz/.h5ad files
- `singlet.read_1pz(path)` — read .1pz sparse matrix format
- `singlet.write_1pz(adata, path)` — write .1pz format
- `singlet.read_kraken2(path)` — read microbiome matrix

### Sample Index Fields
`gsm_id`, `gse_id`, `organism`, `status`, `protocol`, `mapping_rate`, `cells_called`, `median_genes`, `median_umis`, `mt_pct`, `doublet_rate`, `wall_time_s`, `title`

### load_dir() Output
- **obs**: total_umis, total_genes, mt_pct, ribo_pct, intronic_pct, doublet_score, is_doublet, phase, s_score, g2m_score
- **uns**: ancestry, sex_call, summary, saturation_curve, singlify_dir
- **var**: gene_id (Ensembl), gene_name
- **Layers**: gene_counts (default), exon_counts, intron_counts, gene_counts_em

### Bug Fixes
- `top_series()` no longer crashes when `median_genes` column is absent
- `summary()` correctly reports 7 species (was counting multi-species entries as separate)
- Quality tier filtering handles missing QC columns gracefully

### Atlas Stats
- 2,378 samples processed
- 989 successful (42%)
- 1,175 GEO series
- 7 species (human, mouse, macaque, fruit fly, chicken, zebrafish, chimpanzee)
- 2.9M total cells
