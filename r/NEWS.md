# singlet (R) — NEWS

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
