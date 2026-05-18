# Changelog

All notable changes to the singlet project.

## [2.0.0] — Unreleased

### Breaking Changes
- **Unified package**: `singlet-bio`, `singlepress`, `singlet-gpu` merged into single `singlet` package
- **Install**: `pip install singlet` (replaces `pip install singlet-bio`)
- **R package**: renamed from `singlify` to `singlet`
- **C++ namespaces**: `singlet::pz`, `singlet::fq`, `singlet::pileup`, `singlet::gpu`
- **License unified to MIT** across all first-party code. The former GPU library (ex-`singlet-gpu`) was relicensed from GPL-2.0-or-later to MIT; SPDX identifiers added to all 1,067 first-party source files. Vendored `include/singlet/star/` (STAR — MIT; SIMDe — CC0-1.0) retains its own licenses.

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
- **GPU library** (merged from the former standalone `singlet-gpu` repo, developed over ~162 cycles; full per-cycle record in `state/gpu/cycle-log.md`):
  - `core/sparse_eigensolver.h` — header-only LOBPCG for top-K exterior eigenvalues of sparse symmetric CSR (cuRAND Philox + cuBLAS + cuSPARSE SpMM + cuSOLVER); replaces the n²-dense path in `embed/diffmap`/`embed/dpt` (n=10k: ~8 MB vs 400 MB; n=1M now feasible)
  - Native GPU linear-algebra kernels (~2,500 LOC CUDA) — `core/{types,handles,memory}.h`, `reduce/svd/{deflation,randomized,auto_select}.h`, `reduce/nmf/{fit,cv,chunked}.h` — replaced the factornet runtime dependency
  - 71 C++ correctness tests under `tests/cpp/gpu/`, 49 perf benchmark drivers under `bench/`
  - Frontier kernels vs SOTA: pz_device_loader 6.4× anndata-gpu; lognorm 370× scanpy; hvg pearson-residuals 12,609× scanpy; svd-deflation 27×; nmf 1.82–8.66× sklearn; qc/metrics 429× scanpy; de/wilcoxon up to 388.8× scanpy; de/ttest 8.4–10.4× scanpy
  - `FACTORNET_INCLUDE_DIR` is now an optional migration safety-valve; 14+ deferred-scope binding modules gated behind `SINGLET_GPU_BUILD_DEFERRED` (default OFF)
- **IO schema v2 support**: Loader auto-detects singlify v2 subdirectory layout (`donor/snp_ad.1pz`)
- **MCP server tests**: 25 unit tests covering all parquet-backed tools + call_tool router
- **Python lint**: ruff check + ruff format enforced (0 errors, CI job added)
- **Type annotations**: All public functions annotated with return types
- **Makefile**: `make test`, `make lint`, `make build`, `make pipeline`, `make clean`
- **Python API reference**: `docs/api/python.md` covers all public modules
- **Annotation tests**: 15 unit tests for `_annotate.py` (gene_programs, project, annotate)
- **Security**: `SECURITY.md` vulnerability reporting policy + pip-audit CI job
- **Python tests**: 2024 total (up from 205 at cycle 184, 98% core coverage)
- **Public API**: 168 functions exported
- **Preprocessing API**: `describe()`, `filter_cells()`, `filter_genes()`, `normalize()`, `highly_variable_genes()`, `highly_variable_genes_seurat_v3()`, `scale()`, `subsample()`, `concatenate()`, `calculate_qc_metrics()`, `regress_out()`, `downsample_counts()`, `recipe_seurat()`, `recipe_zheng17()` — standard scRNA-seq workflow without scanpy
- **Analysis pipeline**: `pca()`, `neighbors()`, `spatial_neighbors()`, `leiden()`, `louvain()`, `umap()`, `tsne()`, `diffmap()`, `dpt()`, `harmony()`, `combat()`, `mnn_correct()`, `paga()`, `ingest()`, `scrublet()`, `embedding_density()`, `correlation_matrix()`, `marker_gene_overlap()` — full dimensionality reduction + clustering + batch correction + trajectory + QC without scanpy
- **Differential expression**: `rank_genes_groups()` with BH-corrected p-values, `rank_genes_groups_df()`, `score_genes()` for gene set activity scoring, `dendrogram()` for hierarchical group ordering, `enrichr()` for pathway enrichment
- **Visualization**: `plot_umap()`, `plot_violin()`, `plot_dotplot()`, `plot_scatter()`, `plot_heatmap()`, `plot_stacked_violin()`, `plot_paga()`, `plot_ranking()`, `rank_genes_groups_dotplot()`, `rank_genes_groups_tracksplot()` — publication-ready matplotlib plots
- **Property-based tests**: 14 hypothesis tests for codec round-trip verification
- **Code deduplication**: `convert.py` thin re-export (was 267-line copy)
- **Lint compliance**: 0 ruff errors (B904, UP037, UP035 all resolved)
- **Type checking**: pyright with 0 errors, 10 warnings (all optional-dep false positives)
- **C++ zero-warning build**: -Wall -Wextra with 0 warnings across all 100 test executables
- **Better error messages**: download 404 → FileNotFoundError with catalog guidance
- **Regex-safe search**: All catalog filters use literal matching; special chars like `(`, `[`, `+` no longer crash
- **CLI entry point**: `python -m singlet` prints atlas summary and usage
- **make check**: single target for lint + typecheck + tests
- **Organism auto-detection**: Gene name capitalization heuristic (ALL CAPS → human, Title Case → mouse)
- **Memory-safe NNLS**: Row-at-a-time projection avoids full .toarray() OOM on large datasets
- **Input validation**: write_1pz/write_spz/project/annotate validate AnnData input; quality_tier/precision checked
- **Path UX**: All path-accepting functions expand `~`; write functions auto-create parent directories
- **Helpful warnings**: `load(genes=[...])` warns for missing genes; `project()` warns on low gene overlap (<50%)
- **Atomic downloads**: Interrupted downloads don't corrupt cache (writes to .part file, renames on success)
- **annotate(inplace=True)**: Stores cell_type, cell_type_confidence in adata.obs, X_nmf in adata.obsm
- **Format conversion tests**: MTX round-trip, to_csc, from_mtx edge cases
- **Preprocessing tests**: protocol detection, FASTQ download, quantify, QC, species, export

### Architecture
- `include/singlet/pz/` — .1pz VOCSC codec (13x compression, 4000+ MB/s decode)
- `include/singlet/fq/` — .1fq 2-bit packed FASTQ codec
- `include/singlet/pileup/` — streaming BAM pileup engine (70+ modules)
- `include/singlet/gpu/` — CUDA analysis kernels
- `include/singlet/star/` — STAR aligner API (vendored, MIT license)

### Removed
- Standalone `singlepress` repository (archived)
- Standalone `singlet-gpu` repository (archived)
- Standalone `singlet-bio` repository (archived)
- Agent configs moved to private `singlet-agents` repo

## [1.0.0] — Unreleased

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
