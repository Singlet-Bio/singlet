# test-preprocess.R
# Tests for lognorm() and hvg() preprocessing functions.
# Design ref: singlet-gpu/state/designs/24-r-wrapper-foundation.md §API surface
#
# Tests:
#   1. lognorm() populates assays(sce)$logcounts
#   2. lognorm() vs scater::logNormCounts CPU (relative error ≤ 1e-5)
#   3. hvg()     populates rowData(sce)$highly_variable
#   4. hvg()     seurat_v3 vs scran::modelGeneVar (Jaccard ≥ 0.95 on top-2000)

# ---------------------------------------------------------------------------
# lognorm tests
# ---------------------------------------------------------------------------

test_that("lognorm populates assays(sce)$logcounts", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_no_gpu()

  sce <- make_tiny_sce()
  sce_norm <- lognorm(sce)

  expect_s4_class(sce_norm, "SingleCellExperiment")
  expect_true("logcounts" %in% SummarizedExperiment::assayNames(sce_norm),
    info = "assays(sce)$logcounts must be present after lognorm()")

  logcounts_mat <- SummarizedExperiment::assay(sce_norm, "logcounts")
  # logcounts should be a dense or sparse matrix (dgCMatrix or dgeMatrix)
  expect_true(is(logcounts_mat, "Matrix") || is.matrix(logcounts_mat),
    info = "logcounts should be a Matrix object")
  expect_equal(dim(logcounts_mat), dim(sce),
    info = "logcounts must have same dimensions as input counts")

  # All values must be finite and non-negative
  logcounts_vals <- as.numeric(logcounts_mat)
  expect_true(all(is.finite(logcounts_vals)),
    info = "logcounts must contain only finite values")
  expect_true(all(logcounts_vals >= 0),
    info = "log1p-normalized values must be non-negative")
})

test_that("lognorm GPU vs scater::logNormCounts CPU — relative error ≤ 1e-5", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("scater")
  skip_if_no_gpu()

  sce <- make_tiny_sce()
  target_sum_val <- 1e4

  # GPU result (singletGpu)
  sce_gpu <- lognorm(sce, target_sum = target_sum_val, size_factors = "median")
  gpu_mat  <- as.matrix(SummarizedExperiment::assay(sce_gpu, "logcounts"))

  # CPU reference: scater::logNormCounts uses size factors from librarySizeFactors
  sce_cpu <- scater::logNormCounts(sce, size.factors = scater::librarySizeFactors(sce),
                                   log = TRUE, pseudo.count = 1,
                                   exprs_values = "counts")
  cpu_mat  <- as.matrix(SummarizedExperiment::assay(sce_cpu, "logcounts"))

  # Relative L2 error over all non-zero CPU entries
  nz_mask   <- cpu_mat > 0
  rel_err   <- sqrt(sum((gpu_mat[nz_mask] - cpu_mat[nz_mask])^2)) /
               sqrt(sum(cpu_mat[nz_mask]^2))

  tolerance <- 1e-5
  expect_lt(rel_err, tolerance,
    info = paste0("lognorm relative L2 error = ", formatC(rel_err, format = "e"),
                  " (tolerance = ", tolerance, ")"))
})

# ---------------------------------------------------------------------------
# hvg tests
# ---------------------------------------------------------------------------

test_that("hvg populates rowData(sce)$highly_variable", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("Matrix")
  skip_if_no_gpu()

  sce <- make_tiny_sce()
  sce_norm <- lognorm(sce)
  sce_hvg  <- hvg(sce_norm, n_top = 200L, flavor = "seurat_v3")

  expect_s4_class(sce_hvg, "SingleCellExperiment")

  rd <- SummarizedExperiment::rowData(sce_hvg)
  expect_true("highly_variable" %in% colnames(rd),
    info = "rowData(sce)$highly_variable must be present after hvg()")
  expect_true("variances_norm" %in% colnames(rd),
    info = "rowData(sce)$variances_norm must be present after hvg()")

  hv_col <- rd$highly_variable
  expect_type(hv_col, "logical")
  expect_equal(length(hv_col), nrow(sce_hvg))

  # Exactly n_top genes marked highly variable (seurat_v3 top-N selection)
  n_selected <- sum(hv_col, na.rm = TRUE)
  expect_equal(n_selected, 200L,
    info = paste0("Expected 200 HVGs (n_top = 200), got ", n_selected))
})

test_that("hvg seurat_v3 GPU vs scran::modelGeneVar CPU — Jaccard ≥ 0.95 on top-2000", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("scran")
  skip_if_no_gpu()

  # Use a slightly larger synthetic matrix so variance estimation is meaningful
  set.seed(0xC0FFEE)
  n_genes <- 500L
  n_cells <- 200L
  dense   <- matrix(rnbinom(n_genes * n_cells, mu = 3, size = 0.5),
                    nrow = n_genes, ncol = n_cells)
  counts  <- Matrix::Matrix(dense, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell", seq_len(n_cells))
  sce     <- SingleCellExperiment::SingleCellExperiment(
               assays = list(counts = counts))

  n_top_hvg <- min(200L, n_genes)   # 40% of 500 — forces overlap signal

  # GPU HVG (singletGpu)
  sce_norm  <- lognorm(sce)
  sce_hvg   <- hvg(sce_norm, n_top = n_top_hvg, flavor = "seurat_v3")
  gpu_hv    <- rownames(sce_hvg)[SummarizedExperiment::rowData(sce_hvg)$highly_variable]

  # CPU reference: scran modelGeneVar → top-N by bio component
  dec_scran  <- scran::modelGeneVar(sce, assay.type = "counts")
  top_scran  <- scran::getTopHVGs(dec_scran, n = n_top_hvg)

  # Jaccard similarity
  intersection <- length(intersect(gpu_hv, top_scran))
  union_size   <- length(union(gpu_hv, top_scran))
  jaccard      <- if (union_size == 0L) 1.0 else intersection / union_size

  tolerance <- 0.95
  expect_gte(jaccard, tolerance,
    info = paste0("HVG Jaccard (GPU seurat_v3 vs scran modelGeneVar) = ",
                  round(jaccard, 4), " (threshold = ", tolerance, ")"))
})
