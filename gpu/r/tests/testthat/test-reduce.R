# test-reduce.R
# Tests for run_pca() and run_nmf() dimensionality-reduction functions.
# Design ref: singlet-gpu/state/designs/24-r-wrapper-foundation.md §API surface
#
# Tests:
#   1. run_pca() populates reducedDim(sce, "PCA")
#   2. run_pca() singular values match BiocSingular::runPCA (rel L2 ≤ 1e-4)
#   3. run_nmf() populates reducedDim(sce, "NMF")
#   4. run_nmf() reconstruction loss is decreasing with more factors

# Fixed seed for reproducibility across all PCA/NMF tests
kSeed <- 0xC0FFEE

# ---------------------------------------------------------------------------
# PCA tests
# ---------------------------------------------------------------------------

test_that("run_pca populates reducedDim(sce, 'PCA')", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_gpu()

  sce <- make_tiny_sce()
  sce_norm <- lognorm(sce)
  n_pcs <- 20L

  sce_pca <- run_pca(sce_norm, n_pcs = n_pcs, layer = "logcounts", seed = kSeed)

  expect_s4_class(sce_pca, "SingleCellExperiment")

  # reducedDim "PCA" must be present
  expect_true("PCA" %in% SingleCellExperiment::reducedDimNames(sce_pca),
    info = "reducedDim(sce, 'PCA') must be present after run_pca()")

  pca_mat <- SingleCellExperiment::reducedDim(sce_pca, "PCA")
  expect_true(is.matrix(pca_mat),
    info = "PCA embedding should be a plain matrix")
  expect_equal(nrow(pca_mat), ncol(sce_pca),
    info = "PCA embedding rows must equal number of cells")
  expect_equal(ncol(pca_mat), n_pcs,
    info = paste0("PCA embedding must have ", n_pcs, " columns (n_pcs)"))

  # Per-PC variance (or variance_ratio) in metadata
  pca_meta <- S4Vectors::metadata(sce_pca)$pca
  expect_false(is.null(pca_meta),
    info = "metadata(sce)$pca must be populated by run_pca()")
  expect_true("variance_ratio" %in% names(pca_meta) || "variance" %in% names(pca_meta),
    info = "metadata(sce)$pca must contain variance or variance_ratio")
})

test_that("run_pca GPU singular values match BiocSingular::runPCA — rel L2 ≤ 1e-4", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("BiocSingular")
  skip_if_not_installed("Matrix")
  skip_if_no_gpu()

  sce <- make_tiny_sce(seed = kSeed)
  sce_norm <- lognorm(sce)
  n_pcs <- 10L

  # GPU PCA (singletGpu — factornet randomized SVD backend)
  sce_pca_gpu <- run_pca(sce_norm, n_pcs = n_pcs, layer = "logcounts",
                         svd_solver = "auto", seed = kSeed)
  pca_gpu_mat  <- SingleCellExperiment::reducedDim(sce_pca_gpu, "PCA")

  # CPU reference: BiocSingular::runPCA with IRLBA algorithm
  logcounts_mat <- SummarizedExperiment::assay(sce_norm, "logcounts")
  pca_cpu <- BiocSingular::runPCA(t(as.matrix(logcounts_mat)),
                                  rank = n_pcs,
                                  BSPARAM = BiocSingular::IrlbaParam(),
                                  center = TRUE, scale = FALSE)
  sv_cpu <- pca_cpu$sdev  # standard deviations = sqrt of eigenvalues

  # Extract per-PC standard deviations from GPU result
  pca_gpu_meta <- S4Vectors::metadata(sce_pca_gpu)$pca
  # Prefer variance_ratio → convert back; fall back to direct variance field
  if (!is.null(pca_gpu_meta$sdev)) {
    sv_gpu <- pca_gpu_meta$sdev[seq_len(n_pcs)]
  } else if (!is.null(pca_gpu_meta$variance)) {
    sv_gpu <- sqrt(pca_gpu_meta$variance[seq_len(n_pcs)])
  } else {
    # Compute from the embedding directly (column norms ÷ sqrt(n-1))
    sv_gpu <- apply(pca_gpu_mat, 2L, function(v) sqrt(sum(v^2) / (length(v) - 1L)))
  }
  sv_cpu <- sv_cpu[seq_len(n_pcs)]

  # Relative L2 on singular values
  rel_err <- sqrt(sum((sv_gpu - sv_cpu)^2)) / sqrt(sum(sv_cpu^2))
  tolerance <- 1e-4
  expect_lt(rel_err, tolerance,
    info = paste0("PCA SV rel L2 error = ", formatC(rel_err, format = "e"),
                  " (tolerance = ", tolerance, ")"))
})

# ---------------------------------------------------------------------------
# NMF tests
# ---------------------------------------------------------------------------

test_that("run_nmf populates reducedDim(sce, 'NMF') and rowData NMF_loadings", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_tiny_sce()
  sce_norm <- lognorm(sce)
  n_factors <- 10L

  sce_nmf <- run_nmf(sce_norm, n_factors = n_factors, loss = "MSE",
                     layer = "logcounts", seed = kSeed)

  expect_s4_class(sce_nmf, "SingleCellExperiment")

  # reducedDim "NMF" (H matrix — cells × factors)
  expect_true("NMF" %in% SingleCellExperiment::reducedDimNames(sce_nmf),
    info = "reducedDim(sce, 'NMF') must be present after run_nmf()")
  h_mat <- SingleCellExperiment::reducedDim(sce_nmf, "NMF")
  expect_equal(nrow(h_mat), ncol(sce_nmf),
    info = "NMF H matrix rows must equal number of cells")
  expect_equal(ncol(h_mat), n_factors,
    info = paste0("NMF H matrix must have ", n_factors, " columns"))
  # NMF is non-negative
  expect_true(all(h_mat >= -1e-9),
    info = "NMF H matrix (cell scores) must be non-negative")

  # rowData NMF_loadings (W matrix — genes × factors, stored as data frame col)
  rd <- SummarizedExperiment::rowData(sce_nmf)
  expect_true("NMF_loadings" %in% colnames(rd),
    info = "rowData(sce)$NMF_loadings must be present after run_nmf()")
  w_mat <- rd$NMF_loadings
  expect_equal(nrow(w_mat), nrow(sce_nmf),
    info = "NMF W matrix rows must equal number of genes")
  expect_equal(ncol(w_mat), n_factors,
    info = "NMF W matrix cols must equal n_factors")
  expect_true(all(as.numeric(w_mat) >= -1e-9),
    info = "NMF W matrix (gene loadings) must be non-negative")
})

test_that("run_nmf reconstruction loss decreases monotonically with more factors", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_no_gpu()

  sce <- make_tiny_sce(seed = kSeed)
  sce_norm <- lognorm(sce)

  # Reconstruction loss must decrease as n_factors increases
  # (more factors → strictly better rank-k approximation under MSE)
  factor_counts <- c(5L, 10L, 20L)
  losses <- vapply(factor_counts, function(k) {
    sce_nmf <- run_nmf(sce_norm, n_factors = k, loss = "MSE",
                       layer = "logcounts", seed = kSeed)
    nmf_meta <- S4Vectors::metadata(sce_nmf)
    # Expect metadata to carry final reconstruction loss
    if (!is.null(nmf_meta$nmf_loss)) return(nmf_meta$nmf_loss)
    # Fallback: compute Frobenius reconstruction error manually
    logcounts_mat <- as.matrix(SummarizedExperiment::assay(sce_norm, "logcounts"))
    h_mat <- SingleCellExperiment::reducedDim(sce_nmf, "NMF")
    w_mat <- as.matrix(SummarizedExperiment::rowData(sce_nmf)$NMF_loadings)
    recon  <- w_mat %*% t(h_mat)
    sqrt(sum((logcounts_mat - recon)^2))
  }, numeric(1L))

  # Each step must strictly decrease (or at worst stay equal — within tolerance)
  for (i in seq_along(factor_counts)[-1L]) {
    expect_lte(losses[i], losses[i - 1L] * 1.001,  # 0.1% tolerance for float noise
      info = paste0("NMF loss should not increase when n_factors grows: ",
                    factor_counts[i - 1L], " factors → loss=", round(losses[i - 1L], 4),
                    "; ", factor_counts[i], " factors → loss=", round(losses[i], 4)))
  }
})
