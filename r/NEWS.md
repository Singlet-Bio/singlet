# singlet (R) — NEWS

## singlet (development version)

### New features

- A cell-line sample's cells carry the line as cell type in `colData` (`cell_type`, `cell_type_ontology_term_id`,
  `cell_type_deepest`): the Cellosaurus name and id from the sample's GEO record, or "cultured cell" (CL:0000010). The
  labeller, trained on primary tissue, is unreliable on cultured lines. `sample_facts = FALSE` keeps its calls.
- `write_s1()` writes a count matrix (dgCMatrix or matrix, genes x cells; `SingleCellExperiment`; Seurat) as a `.s1`
  sample object with one matrix member, with the same encoder and options as `singlet.write_s1()` in Python;
  `read_s1()` reads it back exactly as a dgCMatrix, `SingleCellExperiment` or Seurat object.
- `load()` reads any local `.s1` file that holds one count matrix and no atlas members (a `write_s1()` file).
- Concurrent range requests: `load()`, `singlet_cells()`, `download()` and `read_s1()` take `io_threads`, the number of
  HTTP range requests in flight at once: `"auto"` (default; 8 over HTTP, 1 for a local study or file) or a whole number, and
  1 sends them one at a time. Once a study's manifest is read, `load()` fetches the cells tables and members of
  `io_threads` samples together, then all their matrix chunk runs together; `download()` fetches `io_threads` 32 MiB pieces
  at a time; a long range is split into pieces of at least 8 MiB when there are fewer ranges than connections. The
  connections come from one curl pool per process and are reused. The result does not depend on `io_threads`, and it is a
  separate budget from the decoder's CPU `threads`. Same option and defaults as `singlet.torch.S1Stream` in Python.
- `read_s1()` reads an `http(s)` URL as well as a local file.

### Bug fixes

- A 200 response whose `Content-Range` names exactly the requested bytes is taken as a range answer, as `rclone serve s3`
  sends it; before, it was refused as a server that ignores Range. A 200 without `Content-Range` that is not the whole
  object is still refused.
- An integrity failure while loading a study (a server that ignores Range, a sample object replaced since the manifest)
  stops `load()` with a `singlet_s1_integrity` error, as intended; before, it was caught as an unreadable sample and the
  sample skipped with a warning.

## singlet 1.2.1

- `load()` / `singlet_load()` and `singlet_cells()` add the structured sample facts of the singlet.bio study API to `colData`,
  as factors: `tissue_type` (CELLxGENE), `cell_line` and `cell_line_cellosaurus_id` (Cellosaurus), `tumor_sample_type`
  (GDC/TCGA), `disease` and `disease_ontology_term_id` (MONDO, or `normal` / PATO:0000461), `donor_status` and `sample_kind`.
  They come from each sample's own GEO record (`NA` where it does not say), are fetched once per study and session for a study
  read over HTTP, and are absent offline (`SINGLET_OFFLINE`), for a local study or when the API cannot be reached.
  `sample_facts = FALSE` skips them. Matches the Python package 1.3.0.

## singlet 1.2.0

Prepared for CRAN; matches the Python package 1.2.0 (`pip install singlet`).

### New features

- `singlet_search_cells()` counts and locates cells across the whole atlas by cell type, chromosomal sex, genetic ancestry,
  UMIs and genes, and by sample metadata (tissue, disease, stage, organism, assay, ...), with ontology descendants; it
  reports how much of the atlas each sample-level filter cannot see (`$unmapped`). `singlet_cell_fields()` lists the fields
  and their coverage.
- `singlet_fetch_cells()` loads the cells a search matches; `singlet_cells_slice()` lists them as (GSM, 0-based columns)
  and `singlet_gather()` loads such a list (the form the MCP tools and Python's `atlas.gather()` use).

### Changes

- Zstandard 1.5.6 is bundled (`src/zstd/`, BSD): installing no longer needs a system `libzstd`, only a C++17 compiler.
- `curl` moved from Suggests to Imports: every published study is read by HTTP range requests.
- The release policy strips per-cell mitochondrial DNA (`mts_*`) and doublet (`qc_doublet_score`) columns from every
  published object; the documentation no longer lists them.
- A retired or withheld object (HTTP 410) stops with the not-in-the-atlas message.
- The local cache tree is `<cache>/s1`, as in the Python client (it was `<cache>/store`; old caches are not reused).
- Documentation describes the per-sample `.s1` objects and manifests instead of `.singlet` bundles.

## singlet 1.1.0

### New features

- `load()` / `singlet_load()` and `singlet_cells()` put the per-cell columns of every .s1 sample's cells table in
  `colData`: `cell_type` and `cell_type_ontology_term_id` (the CL label at level <= 2 by default), the deepest call
  (`cell_type_deepest*`), the CL path and per-level confidences, `sex_chrom`, `sex_score`, `ancestry_hancestro`,
  `ancestry_prob`, `called`, `called_ed`, `called_ed_fdr`, and the QC, donor and mtDNA columns (categorical codes
  decoded). New arguments `cell_type_level`, `cell_type_min_score` and `include_unvalidated`. See `?singlet_labels`.
- `singlet_cells()` conditions can use those columns, e.g. `singlet_cells("GSE286650", cell_type == "T cell" & called_ed)`.

### Changes

- The curated GEO sample fields are now `geo_tissue`, `geo_cell_type`, `geo_disease`, `geo_sex` and `geo_age`
  (previously `tissue`, `cell_type`, `disease`, `sex`, `age`), so `colData$cell_type` is the per-cell label.
- `qc_doublet_score` is unvalidated and left out of `colData` unless `include_unvalidated = TRUE` or it is named in
  `columns`.
- Combining several studies whose `colData` columns differ fills the missing columns with `NA`.

## singlet 1.0.0

### New features

- `singlet_load()`, `singlet_find()`, `singlet_find_load()` and
  `singlet_download()`: the same functions as `load()`, `find()`,
  `find_load()` and `download()` under names that do not mask
  `base::load()` or `utils::find()`. The documentation now uses them.
- `load()` and `download()` accept GEO sample accessions (`"GSM..."`). The
  parent series is looked up through the Singlet API
  (`<SINGLET_API_BASE>/gsm/<GSM>`), its bundle is downloaded, and `load()`
  keeps only that sample's cells. Several samples of one study are read
  from a single download. Previously any GSM was an error. Each lookup is
  saved in `gsm_parents.tsv` in the bundle cache directory, and when the API
  cannot be reached the cached bundles are searched for the sample, so a
  GSM whose study is already cached loads offline in a later session.
- Samples with no usable cells (a missing or 0 x 0 count matrix, no called
  cells, or an unreadable member) are skipped with one warning instead of
  failing the whole study. They are listed with the reason in
  `metadata(sce)$skipped_samples` (`obj@misc$skipped_samples` for Seurat).

### Changes

- `SINGLET_DATA_BASE` now names the host, as in the Python client: bundles
  are fetched from `<base>/data/<GSE>/<GSE>.singlet`, default
  `https://data.singlet.bio`. A value ending in `/data` (the old
  convention) still works.
- Called cells are read from `cell_calls.tsv` whether its barcode column is
  named `barcode`, `cb`, `cell_barcode` or `CB`, and `is_cell` is accepted as
  `TRUE`/`FALSE`, `True`/`False` or `1`/`0`. A sample whose cell calls list
  no cells is skipped rather than loaded with every barcode.
  `singlet_raw_counts()` follows the same rules.
- `read_singlet()` extracts only the count matrices and cell calls it needs
  instead of the whole archive.
- When `SingleCellExperiment` (or `Seurat`) is installed but fails to load,
  the error now includes the underlying `loadNamespace()` message along with
  the install command.
- The `SINGLET_MODALITIES` help page moved to `man/modality_registry.Rd`;
  its old file name differed from `singlet_modalities.Rd` only by case,
  which breaks checkouts on case-insensitive file systems.
- Examples use GSE138867 (first sample GSM4120733), GSE146974 and
  GSE128639. The `quickstart` and `interop` vignettes are now filed under
  "Working with raw pipeline outputs".

## singlet 1.0.0

First CRAN-targeted release. Adds the user-facing atlas API built around
`.singlet` bundles and GEO accessions.

### New features

- `load(x, as = c("sce", "seurat"), cache_dir = NULL)` — primary entry
  point. Accepts GEO Series accessions (`"GSE..."`), local `.singlet`
  bundle paths, or a mix, and returns a combined `SingleCellExperiment`
  (default) or `Seurat` object. Accessions are downloaded from
  `https://data.singlet.bio` and cached (honoring `SINGLET_DATA_BASE` and
  `SINGLET_CACHE_DIR`); existing bundles are reused.
- `read_singlet(path)` — read one local `.singlet` bundle (ZIP64) into a
  `SingleCellExperiment`. Sums spliced + unspliced features onto the
  bundle's canonical gene axis, restricts to called cells, and attaches
  per-sample study metadata to `colData`.
- `find(query, level = c("gse", "gsm"), limit = 50L)` — natural-language
  search returning matching accessions, one per Series by default (via the
  Singlet search API; override base with `SINGLET_API_BASE`).
- `find_load(query, ...)` — search then load in one step.

### Changes

- NAMESPACE regenerated from roxygen2: fixed the
  `read_singlify_dir`/`read_singlet_dir` mismatch and dropped the stale
  `read_1pz_sce` export. The `.1pz` codec is now treated as a low-level
  detail; `load()`/`read_singlet()` are the documented entry points.
- DESCRIPTION made CRAN-valid (title, description, maintainer, imports,
  license).

## singlet 0.2.0

Initial release of the R wrapper. Reads singlet pipeline output
directories (`.1pz` files plus per-cell TSV sidecars) into native R
sparse matrices and single-cell analysis objects.

### Architecture

- The `.1pz` decoder lives in a **header-only C++ reader** at
  `inst/include/singlet-pileup/pz_reader.h`. It is byte-identical to
  the Python sister package's reader — both bind to the same source.
- The R surface is a **thin Rcpp wrapper** (~100 LOC) at
  `src/pz_io.cpp` that marshals the C++ `ReadResult` into an R named
  list, plus pure-R glue at `R/read.R`, `R/sce.R`, `R/seurat.R` for
  the high-level converters.

### New features

- `read_1pz(path)` — read a single `.1pz` into a `Matrix::dgCMatrix`
  with `user_kv` and `vt_code` attributes.
- `read_singlet_dir(path, include = NULL, exclude = NULL)` — read a
  whole pipeline output directory into a named list of matrices,
  with `attr(., "user_kv")` carrying the embedded GEO context.
- `as_sce(path, primary_assay = "spliced")` — build a
  `SingleCellExperiment` with:
    - `assays$counts` from the chosen per-gene matrix
    - additional assays for the velocity trio + EM-rescued counts
    - `altExp(., "exon_counts")`, `("intron_counts")`, etc. for
      per-feature matrices on different feature axes
    - `colData` auto-loaded from `cell_qc_metrics.tsv`,
      `cell_cycle_scores.tsv`, `doublet_scores.tsv`, `read_stats.tsv`,
      `ambient_contamination.tsv`
    - `metadata(sce)$singlet` from the embedded GEO context
- `as_seurat(path, primary_assay = "spliced", project = NULL)` —
  build a `Seurat` object with:
    - `RNA` assay from the chosen per-gene matrix
    - `spliced` / `unspliced` / `ambiguous` as additional assays
      (scvelo-compatible naming, so velocity workflows just work)
    - `@meta.data` auto-populated from per-cell sidecars
    - `@misc$singlet` from the embedded GEO context
    - `@project.name` defaulting to `gsm_id`
- `print.singlet_dir` S3 method for human-readable directory dumps.

### Documentation

- `README.md` — install, quickstart, embedded metadata explainer.
- `vignettes/quickstart.Rmd` — end-to-end walk-through of the four
  read entry points + a table of which `.1pz` files contain what.
- `vignettes/interop.Rmd` — `scater`/`scran` standard pipeline,
  Seurat standard pipeline, `velociraptor`/`scVelo` RNA velocity,
  cohort merging, kept-vs-dropped matrix table.
- `inst/INSTALL_NOTES.md` — rationale for linking system `libzstd`
  instead of vendoring it (CRAN size, dup symbols, security), per-
  platform install commands, troubleshooting.

### Tests

- 13 `testthat` cases across three files covering single-file reads,
  directory reads, the empirical velocity-trio reconstruction
  invariant (`gene_counts == spliced + unspliced + ambiguous`), and
  both adapters. Tests skip cleanly if `Rcpp`, `SingleCellExperiment`,
  or `Seurat` are not installed in the test environment.

### CI

- `.github/workflows/R-CMD-check.yml` runs `R CMD check --as-cran`
  on Ubuntu + macOS + Windows × R release + devel. A header-sync
  preflight step copies `pz_reader.h`, `pz_writer.h`, and
  `sparse_accumulator.h` from `singlet/include/` into
  `r/inst/include/singlet-pileup/` before each check, keeping the
  C++ reader as a single source of truth across both wrappers.
