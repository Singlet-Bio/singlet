# singlet (R) — NEWS

## singlet 1.1.0

### New features

- `singlet_load()`, `singlet_find()`, `singlet_find_load()` and
  `singlet_download()`: the same functions as `load()`, `find()`,
  `find_load()` and `download()` under names that do not mask
  `base::load()` or `utils::find()`. The documentation now uses them.
- `load()` and `download()` accept GEO sample accessions (`"GSM..."`). The
  parent series is looked up through the Singlet API
  (`<SINGLET_API_BASE>/gsm/<GSM>`), its bundle is downloaded, and `load()`
  keeps only that sample's cells. Several samples of one study are read
  from a single download. Previously any GSM was an error.
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
