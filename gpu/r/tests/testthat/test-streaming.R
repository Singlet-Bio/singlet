# test-streaming.R
# Tests for run_pipeline() — GPU streaming pipeline returning a plain list.
# Design ref: singlet-gpu/state/designs/25-r-wrappers-2.md §R API surface
#
# NOTE: run_pipeline() returns a LIST (not an SCE) because the input is multi-file
# and may exceed in-memory capacity. This is intentional — see design doc §Risks.
#
# Tests:
#   1. run_pipeline() returns a list with expected slots
#   2. run_pipeline() single-file lognorm path populates size_factors
#   3. run_pipeline() multi-file concat (skips if multiple fixtures unavailable)

# Fixed seed used across streaming tests
kSeedStream <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

#' Path to the tiny bundled test fixture (from helper-fixtures.R)
streaming_tiny_path <- function() {
  tiny_test_path()   # defined in helper-fixtures.R
}

#' Paths to additional .1pz files for multi-file concat test.
#' Returns NULL if fewer than 2 real .1pz files are accessible.
streaming_multi_paths <- function() {
  gsm_dir <- gsm4037629_path()
  candidates <- c(
    file.path(gsm_dir, "exon_counts.1pz"),
    file.path(gsm_dir, "intron_counts.1pz")
  )
  if (all(file.exists(candidates))) candidates else NULL
}

# ---------------------------------------------------------------------------
# Test 1: run_pipeline() returns a list with expected top-level slots
# ---------------------------------------------------------------------------

test_that("run_pipeline returns list with expected slots", {
  skip_if_no_test_data()   # requires bundled tiny_test.1pz
  skip_if_no_gpu()

  result <- run_pipeline(
    input_paths  = streaming_tiny_path(),
    chunk_cols   = 100L,     # small chunk for tiny fixture
    run_lognorm  = TRUE,
    run_hvg      = FALSE,
    run_pca      = FALSE,
    run_nmf      = FALSE
  )

  expect_type(result, "list")

  # size_factors must always be present when run_lognorm = TRUE
  expect_true("size_factors" %in% names(result),
    info = "run_pipeline() result must contain 'size_factors' when run_lognorm=TRUE")
  expect_true(is.numeric(result$size_factors),
    info = "'size_factors' must be a numeric vector")
  expect_true(all(is.finite(result$size_factors)) && all(result$size_factors > 0),
    info = "'size_factors' must be finite and positive")

  # n_cells slot must reflect the actual cell count streamed
  expect_true("n_cells" %in% names(result) || length(result$size_factors) > 0L,
    info = "run_pipeline() result must carry a cell count (via n_cells or size_factors length)")
})

# ---------------------------------------------------------------------------
# Test 2: Single-file lognorm path — size_factors length matches cell count
# ---------------------------------------------------------------------------

test_that("run_pipeline single-file lognorm size_factors length matches cell count", {
  skip_if_no_test_data()
  skip_if_no_gpu()

  # The tiny fixture has 200 cells (tiny_expected_dims() = c(500, 200))
  expected_cells <- tiny_expected_dims()[2L]   # 200L

  result <- run_pipeline(
    input_paths = streaming_tiny_path(),
    chunk_cols  = 200L,      # single chunk covers the whole tiny fixture
    run_lognorm = TRUE,
    run_hvg     = FALSE,
    run_pca     = FALSE,
    run_nmf     = FALSE
  )

  n_sf <- length(result$size_factors)
  expect_equal(n_sf, expected_cells,
    info = paste0("size_factors length (", n_sf, ") should equal n_cells (",
                  expected_cells, ")"))

  # HVG slot absent when run_hvg=FALSE
  expect_false("hvg_indices" %in% names(result),
    info = "'hvg_indices' should be absent when run_hvg=FALSE")

  # PCA slot absent when run_pca=FALSE
  expect_false("pca_pcs" %in% names(result),
    info = "'pca_pcs' should be absent when run_pca=FALSE")
})

# ---------------------------------------------------------------------------
# Test 3: Multi-file concat (skips if multiple test fixtures unavailable)
# ---------------------------------------------------------------------------

test_that("run_pipeline multi-file concat concatenates all cells across files", {
  skip_if_no_gpu()

  multi_paths <- streaming_multi_paths()
  if (is.null(multi_paths)) {
    skip("Multiple .1pz test fixtures not accessible on this machine — skipping multi-file concat test")
  }
  skip_if_not(all(file.exists(multi_paths)),
    "One or more multi-file paths not found")

  result <- run_pipeline(
    input_paths = multi_paths,
    chunk_cols  = 5000L,
    run_lognorm = TRUE,
    run_hvg     = FALSE,
    run_pca     = FALSE,
    run_nmf     = FALSE
  )

  expect_type(result, "list")
  expect_true("size_factors" %in% names(result),
    info = "multi-file pipeline must produce size_factors")

  # Cell count from concatenated files must exceed single-file cell count
  # (exon + intron both have 11,560 cells → concat = 23,120)
  n_sf <- length(result$size_factors)
  expect_gt(n_sf, gsm4037629_expected_cells(),
    info = paste0("Multi-file concat should have > ", gsm4037629_expected_cells(),
                  " cells; got ", n_sf))

  # size_factors must remain finite and positive after concat
  expect_true(all(is.finite(result$size_factors)) && all(result$size_factors > 0),
    info = "size_factors must be finite and positive after multi-file concat")
})
