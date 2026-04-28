# helper-fixtures.R
# Shared fixtures and skip helpers for singletGpu testthat suite.
# Sourced automatically by testthat before test files.

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

#' Skip if no GPU is available (nvidia-smi absent or returns non-zero)
skip_if_no_gpu <- function() {
  if (nzchar(Sys.which("nvidia-smi"))) {
    ret <- system2("nvidia-smi", stdout = FALSE, stderr = FALSE)
    if (ret == 0L) return(invisible(NULL))
  }
  testthat::skip("No GPU available (nvidia-smi absent or non-zero exit)")
}

#' Skip if the bundled tiny_test.1pz fixture is missing
skip_if_no_test_data <- function() {
  if (file.exists(tiny_test_path())) return(invisible(NULL))
  testthat::skip("Bundled test fixture inst/extdata/tiny_test.1pz not found")
}

#' Skip if the real GSM4037629 sample directory is absent
skip_if_no_gsm4037629 <- function() {
  if (dir.exists(gsm4037629_path())) return(invisible(NULL))
  testthat::skip("Real-data path GSM4037629 not accessible on this machine")
}

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

#' Path to the bundled tiny_test.1pz fixture
#' (lives in inst/extdata/ and is included in the installed package)
tiny_test_path <- function() {
  system.file("extdata", "tiny_test.1pz", package = "singletGpu",
              mustWork = FALSE)
}

#' Dimensions expected from the bundled tiny fixture
#' (500 genes × 200 cells; synthetic, fixed-seed 0xC0FFEE)
tiny_expected_dims <- function() c(500L, 200L)

#' Path to the canonical GSM4037629 scRNA sample on Clipper NFS
gsm4037629_path <- function() {
  "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna/GSE127/GSE127918/GSM4037629"
}

#' Expected cell count for GSM4037629 (11,560 cells called by singlify)
gsm4037629_expected_cells <- function() 11560L

# ---------------------------------------------------------------------------
# Lightweight SCE builder for tests that need a pre-built SCE without I/O
# ---------------------------------------------------------------------------

#' Build a minimal SingleCellExperiment from a fixed-seed dense count matrix.
#' Used by preprocess and reduce tests when tiny_test.1pz is unavailable.
#' Dimensions: 500 genes × 200 cells (matches tiny fixture).
make_tiny_sce <- function(seed = 0xC0FFEE) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")

  set.seed(seed)
  n_genes <- 500L
  n_cells <- 200L
  # Sparse negative-binomial counts (mu=2, size=1) — realistic single-cell
  dense <- matrix(
    rnbinom(n_genes * n_cells, mu = 2, size = 1),
    nrow = n_genes, ncol = n_cells
  )
  counts <- Matrix::Matrix(dense, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell", seq_len(n_cells))

  SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts)
  )
}
