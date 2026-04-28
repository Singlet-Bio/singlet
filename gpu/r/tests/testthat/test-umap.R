# test-umap.R
# Tests for run_umap() — GPU UMAP writing results to reducedDim(sce, "UMAP").
# Design ref: singlet-gpu/state/designs/25-r-wrappers-2.md §R API surface
#
# Tests:
#   1. run_umap() writes reducedDim(sce, "UMAP") with correct dimensions
#   2. run_umap() n_components param is honored (2 vs 3 components)
#   3. run_umap() is deterministic with a fixed seed (identical output on two runs)

# Fixed seed for reproducibility across all UMAP tests
kSeedUmap <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: SCE with PCA + neighbors ready for UMAP
# ---------------------------------------------------------------------------

make_sce_for_umap <- function(seed = kSeedUmap) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("Matrix")

  set.seed(seed)
  n_genes <- 500L
  n_cells <- 200L
  dense   <- matrix(rnbinom(n_genes * n_cells, mu = 3, size = 1),
                    nrow = n_genes, ncol = n_cells)
  counts  <- Matrix::Matrix(dense, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell", seq_len(n_cells))
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts)
  )

  # Inject PCA embedding
  pca_mat <- matrix(rnorm(n_cells * 20L), nrow = n_cells, ncol = 20L)
  rownames(pca_mat) <- paste0("Cell", seq_len(n_cells))
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca_mat

  # Pre-compute neighbors (required by run_umap() when use_neighbors=TRUE)
  sce <- neighbors(sce, n_neighbors = 15L, use_dimred = "PCA", seed = seed)
  sce
}

# ---------------------------------------------------------------------------
# Test 1: run_umap() writes reducedDim(sce, "UMAP")
# ---------------------------------------------------------------------------

test_that("run_umap writes reducedDim('UMAP') with cells × n_components shape", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_no_gpu()

  sce      <- make_sce_for_umap()
  n_comp   <- 2L
  sce_umap <- run_umap(sce, n_components = n_comp, seed = kSeedUmap,
                       use_neighbors = TRUE)

  expect_s4_class(sce_umap, "SingleCellExperiment")

  umap_names <- SingleCellExperiment::reducedDimNames(sce_umap)
  expect_true("UMAP" %in% umap_names,
    info = "reducedDim(sce, 'UMAP') must be present after run_umap()")

  umap_mat <- SingleCellExperiment::reducedDim(sce_umap, "UMAP")
  expect_true(is.matrix(umap_mat),
    info = "UMAP embedding should be a plain matrix (dense)")
  expect_equal(nrow(umap_mat), ncol(sce_umap),
    info = "UMAP rows must equal number of cells")
  expect_equal(ncol(umap_mat), n_comp,
    info = paste0("UMAP cols must equal n_components = ", n_comp))

  # All values must be finite (no NaN / Inf from degenerate graph)
  expect_true(all(is.finite(umap_mat)),
    info = "UMAP embedding must contain only finite values")
})

# ---------------------------------------------------------------------------
# Test 2: n_components param is honored
# ---------------------------------------------------------------------------

test_that("run_umap n_components param honored (2 vs 3 components)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_gpu()

  sce <- make_sce_for_umap()

  for (n_comp in c(2L, 3L)) {
    sce_umap <- run_umap(sce, n_components = n_comp, seed = kSeedUmap,
                         use_neighbors = TRUE)
    umap_mat <- SingleCellExperiment::reducedDim(sce_umap, "UMAP")
    expect_equal(ncol(umap_mat), n_comp,
      info = paste0("UMAP should have ", n_comp, " columns when n_components=", n_comp,
                    "; got ", ncol(umap_mat)))
  }
})

# ---------------------------------------------------------------------------
# Test 3: Deterministic with a fixed seed (two runs → identical UMAP)
# ---------------------------------------------------------------------------

test_that("run_umap deterministic with fixed seed (two runs identical within 1e-6)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_gpu()

  sce <- make_sce_for_umap(seed = kSeedUmap)

  sce_umap1 <- run_umap(sce, n_components = 2L, seed = kSeedUmap, use_neighbors = TRUE)
  sce_umap2 <- run_umap(sce, n_components = 2L, seed = kSeedUmap, use_neighbors = TRUE)

  mat1 <- SingleCellExperiment::reducedDim(sce_umap1, "UMAP")
  mat2 <- SingleCellExperiment::reducedDim(sce_umap2, "UMAP")

  # Frobenius distance between the two embeddings should be ~0
  frob_diff <- sqrt(sum((mat1 - mat2)^2))
  tolerance <- 1e-6
  expect_lt(frob_diff, tolerance,
    info = paste0("UMAP embeddings with same seed should be identical; ",
                  "Frobenius diff = ", formatC(frob_diff, format = "e")))
})
