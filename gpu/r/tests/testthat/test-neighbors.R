# test-neighbors.R
# Tests for neighbors() — kNN graph construction writing to metadata(sce)$neighbors.
# Design ref: singlet-gpu/state/designs/25-r-wrappers-2.md §R API surface
#
# Tests:
#   1. neighbors() stores result in metadata(sce)$neighbors (indices + distances + connectivities)
#   2. neighbors() n_neighbors param is reflected in index matrix dimensions
#   3. neighbors() GPU kNN vs scran::buildKNNGraph — Jaccard ≥ 0.95 on shared edges

# Fixed seed used across all neighbors tests
kSeedNeighbors <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: build a pre-processed SCE with a PCA embedding ready for kNN
# ---------------------------------------------------------------------------

make_sce_with_pca <- function(seed = kSeedNeighbors) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
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
  # Inject a synthetic PCA embedding (cells × 20 PCs) so neighbors() has input
  set.seed(seed)
  pca_mat <- matrix(rnorm(n_cells * 20L), nrow = n_cells, ncol = 20L)
  rownames(pca_mat) <- paste0("Cell", seq_len(n_cells))
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca_mat
  sce
}

# ---------------------------------------------------------------------------
# Test 1: neighbors() stores metadata$neighbors
# ---------------------------------------------------------------------------

test_that("neighbors stores metadata$neighbors with indices, distances, connectivities", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce     <- make_sce_with_pca()
  k_val   <- 15L
  sce_knn <- neighbors(sce, n_neighbors = k_val, use_dimred = "PCA", seed = kSeedNeighbors)

  expect_s4_class(sce_knn, "SingleCellExperiment")

  nbrs <- S4Vectors::metadata(sce_knn)$neighbors
  expect_false(is.null(nbrs),
    info = "metadata(sce)$neighbors must be set after neighbors()")

  # Required sub-slots
  expect_true("indices" %in% names(nbrs),
    info = "metadata(sce)$neighbors$indices must be present")
  expect_true("distances" %in% names(nbrs),
    info = "metadata(sce)$neighbors$distances must be present")
  expect_true("connectivities" %in% names(nbrs),
    info = "metadata(sce)$neighbors$connectivities must be present")

  # Basic shape checks on indices matrix (cells × k)
  idx_mat <- nbrs$indices
  expect_equal(nrow(idx_mat), ncol(sce_knn),
    info = "neighbors$indices must have one row per cell")
  expect_equal(ncol(idx_mat), k_val,
    info = paste0("neighbors$indices must have ", k_val, " columns (n_neighbors)"))

  # Distances must be non-negative
  dist_mat <- nbrs$distances
  expect_true(all(as.numeric(dist_mat) >= 0),
    info = "neighbors$distances must be non-negative")
})

# ---------------------------------------------------------------------------
# Test 2: n_neighbors parameter is reflected in index matrix dimensions
# ---------------------------------------------------------------------------

test_that("neighbors n_neighbors matches input (k=10 and k=30 both valid)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_sce_with_pca()

  for (k_val in c(10L, 30L)) {
    sce_knn <- neighbors(sce, n_neighbors = k_val, use_dimred = "PCA", seed = kSeedNeighbors)
    nbrs    <- S4Vectors::metadata(sce_knn)$neighbors

    expect_equal(ncol(nbrs$indices), k_val,
      info = paste0("neighbors$indices cols should equal n_neighbors = ", k_val,
                    "; got ", ncol(nbrs$indices)))
    expect_equal(nrow(nbrs$indices), ncol(sce),
      info = "neighbors$indices rows must equal cell count regardless of k")
  }
})

# ---------------------------------------------------------------------------
# Test 3: GPU kNN vs scran::buildKNNGraph — Jaccard ≥ 0.95 on shared edges
# ---------------------------------------------------------------------------

test_that("neighbors vs scran::buildKNNGraph Jaccard ≥ 0.95", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("scran")
  skip_if_not_installed("igraph")
  skip_if_no_gpu()

  sce   <- make_sce_with_pca(seed = kSeedNeighbors)
  k_val <- 15L

  # GPU kNN (singletGpu)
  sce_knn  <- neighbors(sce, n_neighbors = k_val, use_dimred = "PCA",
                        metric = "euclidean", seed = kSeedNeighbors)
  idx_gpu  <- S4Vectors::metadata(sce_knn)$neighbors$indices  # cells × k

  # CPU reference: scran::buildKNNGraph on the same PCA embedding
  pca_mat   <- SingleCellExperiment::reducedDim(sce, "PCA")
  g_cpu     <- scran::buildKNNGraph(pca_mat, k = k_val, d = ncol(pca_mat),
                                    transposed = FALSE)
  el_cpu    <- igraph::as_edgelist(g_cpu, names = FALSE)  # n_edges × 2

  # Convert GPU indices to an edge set (undirected: sort each pair)
  n_cells  <- nrow(idx_gpu)
  gpu_edges <- do.call(rbind, lapply(seq_len(n_cells), function(i) {
    nbrs_i <- idx_gpu[i, ]
    cbind(pmin(i, nbrs_i), pmax(i, nbrs_i))
  }))
  gpu_set <- unique(paste(gpu_edges[, 1L], gpu_edges[, 2L], sep = "-"))

  cpu_edges_sorted <- cbind(pmin(el_cpu[, 1L], el_cpu[, 2L]),
                            pmax(el_cpu[, 1L], el_cpu[, 2L]))
  cpu_set <- unique(paste(cpu_edges_sorted[, 1L], cpu_edges_sorted[, 2L], sep = "-"))

  # Jaccard on edge sets
  intersection <- length(intersect(gpu_set, cpu_set))
  union_size   <- length(union(gpu_set, cpu_set))
  jaccard      <- if (union_size == 0L) 1.0 else intersection / union_size

  tolerance <- 0.95
  expect_gte(jaccard, tolerance,
    info = paste0("Edge Jaccard (GPU kNN vs scran::buildKNNGraph, k=", k_val, ") = ",
                  round(jaccard, 4), " (threshold = ", tolerance, ")"))
})
