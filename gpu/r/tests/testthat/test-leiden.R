# test-leiden.R
# Tests for leiden() — GPU Leiden clustering writing results to colData(sce).
# Design ref: singlet-gpu/state/designs/25-r-wrappers-2.md §R API surface
#
# Tests:
#   1. leiden() writes a factor column to colData(sce) (default column "leiden")
#   2. leiden() resolution param affects n_clusters (higher resolution → more clusters)
#   3. leiden() GPU vs igraph::cluster_leiden CPU — ARI ≥ 0.85

# Fixed seed for reproducibility across all leiden tests
kSeedLeiden <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: build an SCE with neighbors already computed (required by leiden())
# ---------------------------------------------------------------------------

make_sce_with_neighbors <- function(k = 15L, seed = kSeedLeiden) {
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

  # Run neighbors() so leiden() has graph input
  sce <- neighbors(sce, n_neighbors = k, use_dimred = "PCA", seed = seed)
  sce
}

# ---------------------------------------------------------------------------
# Test 1: leiden() writes colData column as factor
# ---------------------------------------------------------------------------

test_that("leiden writes colData column as factor (default column 'leiden')", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce     <- make_sce_with_neighbors()
  sce_lei <- leiden(sce, resolution = 1.0, seed = kSeedLeiden)

  expect_s4_class(sce_lei, "SingleCellExperiment")

  cd <- SummarizedExperiment::colData(sce_lei)
  expect_true("leiden" %in% colnames(cd),
    info = "colData(sce)$leiden must be present after leiden()")

  leiden_col <- cd$leiden
  expect_true(is.factor(leiden_col),
    info = "colData(sce)$leiden must be a factor")
  expect_equal(length(leiden_col), ncol(sce_lei),
    info = "leiden column length must equal number of cells")

  # At least 2 clusters expected on random data with resolution=1.0
  n_clusters <- nlevels(leiden_col)
  expect_gte(n_clusters, 2L,
    info = paste0("Expected ≥2 Leiden clusters; got ", n_clusters))
})

# ---------------------------------------------------------------------------
# Test 2: resolution param affects n_clusters (higher → more clusters)
# ---------------------------------------------------------------------------

test_that("leiden resolution param affects n_clusters (higher resolution → more clusters)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_no_gpu()

  sce <- make_sce_with_neighbors()

  res_low  <- 0.2
  res_high <- 2.0

  sce_low  <- leiden(sce, resolution = res_low,  seed = kSeedLeiden)
  sce_high <- leiden(sce, resolution = res_high, seed = kSeedLeiden)

  n_low  <- nlevels(SummarizedExperiment::colData(sce_low)$leiden)
  n_high <- nlevels(SummarizedExperiment::colData(sce_high)$leiden)

  expect_gte(n_high, n_low,
    info = paste0("Higher resolution (", res_high, ") should produce ≥ clusters than low (",
                  res_low, "); got n_low=", n_low, " n_high=", n_high))
})

# ---------------------------------------------------------------------------
# Test 3: GPU Leiden vs igraph::cluster_leiden — ARI ≥ 0.85
# ---------------------------------------------------------------------------

test_that("leiden vs igraph::cluster_leiden ARI ≥ 0.85", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("igraph")
  skip_if_not_installed("mclust")   # for adjustedRandIndex
  skip_if_no_gpu()

  sce        <- make_sce_with_neighbors(k = 15L, seed = kSeedLeiden)
  resolution <- 1.0

  # GPU Leiden (singletGpu)
  sce_lei    <- leiden(sce, resolution = resolution, seed = kSeedLeiden)
  gpu_labels <- as.integer(SummarizedExperiment::colData(sce_lei)$leiden)

  # CPU reference: build igraph from the same connectivities matrix and cluster
  conn_mat <- S4Vectors::metadata(sce)$neighbors$connectivities
  # connectivities is expected to be a sparse symmetric matrix (dgCMatrix)
  g_cpu    <- igraph::graph_from_adjacency_matrix(
    conn_mat, mode = "undirected", weighted = TRUE, diag = FALSE
  )
  set.seed(kSeedLeiden)
  cpu_result <- igraph::cluster_leiden(
    g_cpu,
    objective_function = "modularity",
    resolution_parameter = resolution
  )
  cpu_labels <- igraph::membership(cpu_result)

  # Adjusted Rand Index
  ari <- mclust::adjustedRandIndex(gpu_labels, as.integer(cpu_labels))
  tolerance <- 0.85
  expect_gte(ari, tolerance,
    info = paste0("ARI (GPU Leiden vs igraph::cluster_leiden) = ",
                  round(ari, 4), " (threshold = ", tolerance, ")"))
})
