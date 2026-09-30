# Changelog

All notable changes to the singlet project.

## [Unreleased]

## [2.1.0] — 2026-10-01

The first release published to PyPI, as **`singlet-bio`** (`pip install singlet-bio`, then `import singlet`). It collects the client fixes from PR #2:

- **Distribution renamed to `singlet-bio`**; the import name (`singlet`) and the R package name are unchanged.
- **`singlet.find()` returns studies (`GSE`) by default**, not samples; pass `level="gsm"` for the 2.0.0 behaviour.
- **Hollow-sample guards** end to end: loading skips empty or 0x0 samples and explains all-hollow studies, `pack_gse` refuses hollow or failed-write samples, and the pipeline no longer writes 0x0 stubs for called cells.
- **`singlet-mcp`** is the local MCP server's console script; `singlet` remains as a deprecated alias that no longer shadows the pipeline binary.
- **Functions whose hosts were never in service raise** a clear `NotImplementedError` pointing at `singlet.load` / `singlet.find`, instead of failing against dead hosts.

### Packaging
- **The Python distribution is now `singlet-bio`.** `singlet` on PyPI belongs to an unrelated 2017 project, so this package will be published as `singlet-bio` (this reverses the 2.0.0 "`pip install singlet`" plan). The import name is unchanged (`import singlet`), as is the R package name (`singlet`). Until the first release, install with `pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"`; after it, `pip install singlet-bio`. Extras follow the new name (`singlet-bio[analysis]`, `singlet-bio[mcp]`, ...), and wheels/sdists are named `singlet_bio-*`. Version lookups read the `singlet-bio` distribution first and fall back to `singlet` for installs made before the rename.

### Python — fixes
- **h5ad / zarr export works for real studies.** `SingletBundle.to_anndata()` (and so `singlet.load()`) no longer copies `manifest.json` into `adata.uns` verbatim: its `checksums` were keyed by archive paths containing `/`, which `write_h5ad`/`write_zarr` reject, so every real study failed to export. `uns["manifest"]["checksums"]` is now parallel `path`/`sha256` lists, and `uns` values are made writable (no `/` in keys, mixed lists and lists of records stored as JSON text, empty lists dropped).
- **Empty samples are skipped, hollow studies are explained.** Samples whose count matrix is a 0x0 stub, missing, or has no called cells are skipped with a warning and listed in `adata.uns["skipped_samples"]` (`gsm_id`, `reason`). A study with no usable sample raises an error that says the file is hollow and links to `https://singlet.bio/study/<GSE>`; loading a skipped GSM says why.
- **Called cells.** Bundle readers accept every barcode-column spelling in `cell_calls.tsv` (`barcode`, `cb`, `cell_barcode`, `CB`) and honour `is_cell` written as bool, 0/1 or text (the string `"False"` no longer counts as a cell).
- **`load(genes=[...])` accepts gene symbols** as well as Ensembl ids: exact `var["gene_name"]` match first, then case-insensitive; unmatched names are warned about. `var_names` stay Ensembl ids.
- **`pack_gse` refuses hollow samples.** A sample whose `summary.json` reports called cells while `exon_counts.1pz` is missing, a 0x0 stub (read from the TP1Z header) or all zeros is excluded with a warning and recorded in `manifest.json` under `excluded_samples`; `strict=True` / `--strict` raises instead, and a study that is entirely hollow is never packed.
- **`pack_gse` refuses samples whose matrices failed to write.** A sample whose `summary.json` has `status: "fail_export_matrix"` or a `write_failed:<file>` warning (for example `intron_counts.1pz` failed while `exon_counts.1pz` was fine) is excluded and listed in `excluded_samples`, instead of shipping exon-only counts that are not comparable with its siblings'.
- **`adata.uns["study_meta"]` keeps keys whose value is `None`** (e.g. `gsm_meta[<GSM>]["mapping_rate"]`), as before; they are no longer dropped, since anndata writes `None` natively. Lists of records such as `publications` are still stored as JSON text so the AnnData can be written (`json.loads(adata.uns["study_meta"]["publications"])` returns the records; `SingletBundle.study_meta` is the raw dict), and empty lists are still dropped (HDF5 cannot compress a zero-length dataset).
- **`fetch()` / `open("GSM…")` honour a configured sample mirror again.** The mirror is read from the new `$SINGLET_SAMPLE_MIRROR`; a value in `$SINGLET_DATA_BASE` (what `fetch()` read up to 2.0.0; it now names the bundle host) is still used for it with a `FutureWarning`. `default_base_url()` again returns the sample-mirror base (the retired `https://data.singlet.bio/v1` when none is set, which `fetch()` refuses with a clear `NotImplementedError` rather than a 404).
- **`manifest.json` records the real `singlet_version`** when packing from a source checkout (it read a `__version__` that `singlet._versions` never defined and fell back to `"unknown"`).
- **`$SINGLET_DATA_BASE`** means one thing: the base URL such that bundles live at `<base>/data/<GSE>/<GSE>.singlet`. A trailing `/data` (the R client's form) is accepted.
- **Pipeline binary discovery** looks for `singlet-pipeline` first and skips a `singlet` on `$PATH` that is a Python console script, so the MCP alias can no longer shadow the C++ binary.
- `load_sample()` raises a clear error when no local catalog is configured, before touching the compiled codec; it is kept for cluster users but removed from the public API listing and docs.

### Python — changes
- **`singlet.find()` returns study (`GSE`) accessions by default (`level="gse"`); 2.0.0 returned samples (`GSM`).** There is no runtime warning, so check callers: code that matches the result against sample ids (`adata.obs["gsm_id"].isin(singlet.find(q))`) must pass `level="gsm"`, and `singlet.load(singlet.find(q))` now downloads every matching study in full — slice it (`singlet.find(q)[:3]`). `find_load()` loads the top **3** studies by default. The local MCP server's `singlet_nl_search` tool defaults to `level="gse"` as well.
- `singlet.info()` falls back to the live API (`/api/gse/<GSE>`, `/api/gsm/<GSM>`) for accessions missing from the bundled offline snapshot; `catalog()`/`summary()` are documented as that snapshot. `python -m singlet` prints live totals from `https://singlet.bio/api/stats` (snapshot summary when offline). `SINGLET_OFFLINE=1` disables these live lookups.
- The local MCP server's console script is now **`singlet-mcp`**; `singlet` remains as a deprecated alias.
- Extras: `mcp` pinned to `mcp>=1.0,<2`; new `analysis` extra (matplotlib, scanpy, statsmodels, igraph, leidenalg); new `test` extra; `dev` pulls `test`.
- **Retired** (raise `NotImplementedError` with a pointer to `singlet.load`/`singlet.find`), because their hosts were never in service: `gene_programs`/`project`/`annotate` (models.singlet.bio), `query`/`search`/`login` and hosted NMF serving (api.singlet.bio/v1), `fetch()` with no sample mirror configured (data.singlet.bio/v1), and `singlet.atlas` (r2.singlet.bio).
- **`singlet.login()` now raises** instead of silently storing a key nothing used. Scripts that call it before `singlet.load()` stop at that line; the error says to delete the call (downloads need no key) or, for the natural-language search key, to use `singlet.set_api_key(key)` / `$SINGLET_API_KEY`. The key is never echoed.
- Docs: install from GitHub (`pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"`, needs a C++17 compiler and zstd) until wheels are published; examples use GSE138867 / GSM4120733; stale counts and non-existent GPU/loom/`singlet[pipeline]` instructions removed.

### Pipeline (C++)
- A `.1pz` output that fails to write is recorded in `summary.json` (`status: "fail_export_matrix"`, warning `write_failed:<file>`) and gets no 0x0 stub; `pack_gse` then leaves the sample out. A `.1pz` run that calls cells but leaves no `exon_counts.1pz` is recorded the same way (`exon_counts_1pz_missing`).
- `--output-format mtx|h5ad|loom` runs are not failures: they write no `.1pz` by design, so they keep their status, get a `no_1pz_output_format` warning and the usual `exon_counts.1pz` stub (which `pack_gse` refuses as hollow when cells were called).

### Development
- The CI lint job passes: `python/` and `tests/python/` are `ruff check` / `ruff format` clean, and ruff (0.16.9) and pyright (1.1.414) are pinned in `ci.yml` so a new release cannot turn a green branch red. Pushes to `claude/lint-*` branches also run `autofix.yml`, which commits `ruff check --fix` + `ruff format` back to the branch.
- Real bugs the lint pass turned up: the GPU test modules called `singlet.gpu.*` without importing it; `singlet.gpu.preprocess.lognorm._prepare_adata` called the `copy` module instead of copying; a duplicate test-class name in `test_convert.py` hid three `singlet.convert` tests, which now run.

## [2.0.0] — 2026-09-04

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
