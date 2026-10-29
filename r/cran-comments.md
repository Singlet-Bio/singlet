## Submission

This is a new submission: the first CRAN release of 'singlet' (version 1.2.1), the R client for the Singlet single-cell RNA-seq
atlas (<https://singlet.bio>). The matching Python package is 'singlet' on PyPI (1.2.0; 1.3.0 adds the same sample facts).

## Test environments

GitHub Actions, `R CMD check --as-cran --no-manual` (`.github/workflows/R-CMD-check.yml`, run on PR #72):

- ubuntu-latest, R release
- ubuntu-latest, R devel
- ubuntu-latest, R oldrel-1
- macos-latest, R release
- windows-latest, R release

Also `R CMD check --as-cran --run-donttest` with CRAN's incoming checks (`_R_CHECK_CRAN_INCOMING_REMOTE_=true`) on
Ubuntu 24.04, R 4.4 (Bioconductor 3.20 image): incoming feasibility reported only "New submission"; tests and examples
passed. Its other notes came from that image (no Seurat or qpdf installed, its own compiler flags, no clock check).

To add before submitting: win-builder R-devel (`devtools::check_win_devel()`) and the CRAN macOS builder
(`devtools::check_mac_release()`).

## R CMD check results

0 errors | 0 warnings | 0 notes on all five GitHub Actions environments. CRAN's incoming checks will add:

* checking CRAN incoming feasibility ... NOTE
  New submission

The `download()` example (a 0.2 Mb study, fetched into `tempdir()`) can take a few seconds over a slow network.

On Linux the installed size is about 11 Mb, almost all of it the compiled library with debugging symbols (the source
tarball is 0.6 Mb); a "sub-directories of 1Mb or more: libs" NOTE may appear on some platforms.

## Notes for the CRAN team

- The package bundles Zstandard 1.5.6 (`src/zstd/`, BSD licence) so that it needs no system library on any platform. The
  copyright holders are listed in Authors@R and in `inst/COPYRIGHTS`; the multithreaded compressor, dictionary builder,
  legacy formats and assembly decoder are left out.
- Functions that query the atlas need network access. Their examples are in `\donttest{}`, guarded by
  `curl::has_internet()`, and fetch one small sample (a few MB) into `tempdir()`. Examples left in `\dontrun{}` need a
  local pipeline output directory or `.singlet` file that cannot be shipped, an API key (`set_api_key()`), or the
  rate-limited natural-language search (`find()`, `find_load()`). Tests that use the network are skipped on CRAN; the
  remaining tests run offline.
- Downloads are cached in `tools::R_user_dir("singlet", "cache")` (overridable with `SINGLET_CACHE_DIR`); tests and examples
  write only to `tempdir()`.
- `load()` and `find()` mask `base::load()` and `utils::find()` when the package is attached. Prefixed aliases
  (`singlet_load()`, `singlet_find()`, ...) are exported and used throughout the documentation.

## Reverse dependencies

None (new package).
