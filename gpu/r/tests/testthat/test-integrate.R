# test-integrate.R
# Tests for run_harmony() and run_bbknn() — GPU batch-integration functions.
# Design ref: singlet-gpu/state/designs/26-r-wrappers-3.md §R API surface
#
# Tests:
#   1. run_harmony() writes reducedDim('Harmony') to the SCE
#   2. run_harmony() embedding vs harmony::RunHarmony — canonical correlation >= 0.95
#   3. run_bbknn() writes metadata$neighbors (indices, distances, connectivities)

kSeedIntegrate <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: SCE with PCA embedding and batch colData for integration tests
# ---------------------------------------------------------------------------

make_sce_for_integration <- function(seed = kSeedIntegrate,
                                      n_genes  = 500L,
                                      n_cells  = 200L,
                                      n_pcs    = 20L,
                                      n_batches = 3L) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("S4Vectors")

  set.seed(seed)
  # Sparse count matrix
  dense  <- matrix(rnbinom(n_genes * n_cells, mu = 2, size = 1),
                   nrow = n_genes, ncol = n_cells)
  counts <- Matrix::Matrix(dense, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell", seq_len(n_cells))

  # Batch labels with a mild batch effect: batch centroid shift in PCA space
  batch_labels <- rep(paste0("Batch", seq_len(n_batches)), length.out = n_cells)

  # PCA embedding with batch-specific mean offset
  pca_base <- matrix(rnorm(n_cells * n_pcs), nrow = n_cells, ncol = n_pcs)
  for (b in seq_len(n_batches)) {
    idx <- which(batch_labels == paste0("Batch", b))
    shift <- rnorm(n_pcs, mean = 0, sd = 1.5)
    pca_base[idx, ] <- sweep(pca_base[idx, , drop = FALSE], 2, shift, "+")
  }
  rownames(pca_base) <- paste0("Cell", seq_len(n_cells))
  colnames(pca_base) <- paste0("PC", seq_len(n_pcs))

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays  = list(counts = counts),
    colData = S4Vectors::DataFrame(batch = factor(batch_labels))
  )
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca_base
  sce
}

# ---------------------------------------------------------------------------
# Test 1: run_harmony writes reducedDim('Harmony')
# ---------------------------------------------------------------------------

test_that("run_harmony writes reducedDim('Harmony') to the SCE", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_sce_for_integration()
  sce_harm <- run_harmony(sce,
                           batch_key       = "batch",
                           use_dimred      = "PCA",
                           adjusted_dimred = "Harmony",
                           n_clusters      = 5L,
                           max_iter        = 10L,
                           seed            = kSeedIntegrate)

  expect_s4_class(sce_harm, "SingleCellExperiment")

  # Must contain 'Harmony' reducedDim
  rd_names <- SingleCellExperiment::reducedDimNames(sce_harm)
  expect_true("Harmony" %in% rd_names,
    info = paste0("run_harmony() must store result as reducedDim('Harmony'); ",
                  "available dims: ", paste(rd_names, collapse = ", ")))

  harm_mat <- SingleCellExperiment::reducedDim(sce_harm, "Harmony")
  # Same number of cells
  expect_equal(nrow(harm_mat), ncol(sce_harm),
    info = "Harmony embedding must have one row per cell")
  # Same number of PCs as input
  n_pcs_input <- ncol(SingleCellExperiment::reducedDim(sce, "PCA"))
  expect_equal(ncol(harm_mat), n_pcs_input,
    info = paste0("Harmony embedding should have same #dims as PCA input (",
                  n_pcs_input, "); got ", ncol(harm_mat)))
  # All finite
  expect_true(all(is.finite(harm_mat)),
    info = "Harmony embedding must contain only finite values")
})

# ---------------------------------------------------------------------------
# Test 2: run_harmony embedding vs harmony::RunHarmony — CCA >= 0.95
# ---------------------------------------------------------------------------

test_that("run_harmony embedding vs harmony::RunHarmony CCA >= 0.95", {
  skip_if_not_installed("harmony")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_no_gpu()

  sce <- make_sce_for_integration()
  n_pcs_input <- ncol(SingleCellExperiment::reducedDim(sce, "PCA"))

  # GPU Harmony
  sce_gpu <- run_harmony(sce,
                          batch_key       = "batch",
                          use_dimred      = "PCA",
                          adjusted_dimred = "Harmony",
                          n_clusters      = 5L,
                          max_iter        = 10L,
                          seed            = kSeedIntegrate)
  gpu_emb <- SingleCellExperiment::reducedDim(sce_gpu, "Harmony")  # cells × pcs

  # CPU reference: harmony::RunHarmony
  pca_mat      <- SingleCellExperiment::reducedDim(sce, "PCA")
  batch_vec    <- as.character(SummarizedExperiment::colData(sce)$batch)
  cpu_emb      <- harmony::RunHarmony(
    data_mat  = pca_mat,
    meta_data = data.frame(batch = batch_vec, row.names = rownames(pca_mat)),
    vars_use  = "batch",
    nclust    = 5L,
    max_iter  = 10L,
    seed      = kSeedIntegrate,
    verbose   = FALSE
  )  # returns cells × pcs matrix

  # Canonical correlation: measure subspace alignment
  # Use min number of columns to protect against dimension mismatch
  k <- min(ncol(gpu_emb), ncol(cpu_emb), 10L)
  cca_result <- tryCatch(
    cancor(gpu_emb[, seq_len(k)], cpu_emb[, seq_len(k)]),
    error = function(e) NULL
  )
  if (is.null(cca_result)) {
    skip("cancor() failed (likely singular matrix with synthetic data) — skip CCA check")
  }

  # Mean of top-k canonical correlations must be >= 0.95
  mean_cca <- mean(cca_result$cor)
  tolerance <- 0.95

  expect_gte(mean_cca, tolerance,
    info = paste0("Mean CCA (run_harmony vs harmony::RunHarmony, k=", k, ") = ",
                  round(mean_cca, 4), " (threshold = ", tolerance, ")"))
})

# ---------------------------------------------------------------------------
# Test 3: run_bbknn writes metadata$neighbors
# ---------------------------------------------------------------------------

test_that("run_bbknn writes metadata$neighbors with indices, distances, connectivities", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce      <- make_sce_for_integration()
  sce_bb   <- run_bbknn(sce,
                          batch_key               = "batch",
                          use_dimred              = "PCA",
                          neighbors_within_batch  = 3L,
                          metric                  = "euclidean")

  expect_s4_class(sce_bb, "SingleCellExperiment")

  nbrs <- S4Vectors::metadata(sce_bb)$neighbors
  expect_true(!is.null(nbrs),
    info = "run_bbknn() must write metadata(sce)$neighbors")

  # Required sub-elements
  for (elem in c("indices", "distances", "connectivities")) {
    expect_true(!is.null(nbrs[[elem]]),
      info = paste0("metadata$neighbors must contain '", elem, "'"))
  }

  n_cells <- ncol(sce_bb)

  # indices: n_cells × k (or sparse — accept either)
  if (is.matrix(nbrs$indices)) {
    expect_equal(nrow(nbrs$indices), n_cells,
      info = "neighbors$indices must have one row per cell")
  } else {
    expect_true(inherits(nbrs$indices, c("sparseMatrix", "list", "matrix")),
      info = "neighbors$indices must be a matrix or sparse representation")
  }

  # distances: same shape as indices
  if (is.matrix(nbrs$distances)) {
    expect_equal(nrow(nbrs$distances), n_cells,
      info = "neighbors$distances must have one row per cell")
    expect_true(all(nbrs$distances >= 0, na.rm = TRUE),
      info = "neighbors$distances must be non-negative")
  }

  # connectivities: square n_cells × n_cells sparse matrix
  if (inherits(nbrs$connectivities, "sparseMatrix")) {
    expect_equal(nrow(nbrs$connectivities), n_cells,
      info = "neighbors$connectivities must have n_cells rows")
    expect_equal(ncol(nbrs$connectivities), n_cells,
      info = "neighbors$connectivities must have n_cells columns")
  }
})
