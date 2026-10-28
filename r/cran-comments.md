## Submission

This is a new submission: the first CRAN release of 'singlet' (version 1.2.0), the R client for the Singlet single-cell RNA-seq
atlas (<https://singlet.bio>). The matching Python package is on PyPI as 'singlet' 1.2.0.

## Test environments

- GitHub Actions (`.github/workflows/R-CMD-check.yml`, `R CMD check --as-cran --no-manual`):
  ubuntu-latest R release, ubuntu-latest R devel, macos-latest R release, windows-latest R release.
- Before submitting, also run win-builder (`devtools::check_win_devel()`) and the macOS builder
  (`devtools::check_mac_release()`); results to be added here.

## R CMD check results

See the PR's R CMD check runs; expected: 0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  New submission

## Notes for the CRAN team

- The package bundles Zstandard 1.5.6 (`src/zstd/`, BSD licence) so that it needs no system library on any platform. The
  copyright holders are listed in Authors@R and in `inst/COPYRIGHTS`; the multithreaded compressor, dictionary builder,
  legacy formats and assembly decoder are left out.
- Functions that query the atlas need network access. Their examples are in `\donttest{}` and guarded by
  `curl::has_internet()`; tests that use the network are skipped on CRAN, and the remaining tests are offline.
- Downloads are cached in `tools::R_user_dir("singlet", "cache")` (overridable with `SINGLET_CACHE_DIR`); tests and examples
  write only to `tempdir()`.
- `load()` and `find()` mask `base::load()` and `utils::find()` when the package is attached. Prefixed aliases
  (`singlet_load()`, `singlet_find()`, ...) are exported and used throughout the documentation.

## Reverse dependencies

None (new package).
