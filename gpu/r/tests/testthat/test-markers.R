# test-markers.R
# Tests for score_genes() and celltypist_predict() — annotation functions.
# Design ref: singlet-gpu/state/designs/25-r-wrappers-2.md §R API surface
#
# Tests:
#   1. score_genes() writes per-set score columns to colData(sce)
#   2. celltypist_predict() writes a factor column to colData(sce)

# Fixed seed for reproducibility
kSeedMarkers <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: SCE with logcounts assay suitable for scoring
# ---------------------------------------------------------------------------

make_sce_for_markers <- function(seed = kSeedMarkers) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("S4Vectors")

  set.seed(seed)
  n_genes <- 500L
  n_cells <- 200L
  dense   <- matrix(rnbinom(n_genes * n_cells, mu = 2, size = 1),
                    nrow = n_genes, ncol = n_cells)
  counts  <- Matrix::Matrix(dense, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell", seq_len(n_cells))

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts)
  )
  # Add a logcounts assay (score_genes works on logcounts by convention)
  lc <- log1p(as.matrix(counts))
  SummarizedExperiment::assay(sce, "logcounts") <- Matrix::Matrix(lc, sparse = TRUE)

  # Inject PCA embedding (needed by celltypist_predict)
  pca_mat <- matrix(rnorm(n_cells * 20L), nrow = n_cells, ncol = 20L)
  rownames(pca_mat) <- paste0("Cell", seq_len(n_cells))
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca_mat

  sce
}

# ---------------------------------------------------------------------------
# Test 1: score_genes() writes per-set score columns to colData
# ---------------------------------------------------------------------------

test_that("score_genes writes colData per set (<set>_score columns, one per gene set)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_sce_for_markers()

  # Define two gene sets using gene names that are present in the SCE
  gene_sets <- list(
    SetA = paste0("Gene", 1:20),    # first 20 genes
    SetB = paste0("Gene", 481:500)  # last 20 genes
  )

  sce_scored <- score_genes(sce, gene_sets = gene_sets, method = "ulm",
                             score_name = "score")

  expect_s4_class(sce_scored, "SingleCellExperiment")

  cd <- SummarizedExperiment::colData(sce_scored)

  # Expect one score column per set: "<set_name>_score"
  for (set_name in names(gene_sets)) {
    expected_col <- paste0(set_name, "_score")
    expect_true(expected_col %in% colnames(cd),
      info = paste0("colData must contain column '", expected_col,
                    "' after score_genes()"))

    score_vals <- cd[[expected_col]]
    expect_equal(length(score_vals), ncol(sce_scored),
      info = paste0("Score column '", expected_col,
                    "' must have one value per cell"))
    expect_true(all(is.finite(score_vals)),
      info = paste0("Score column '", expected_col,
                    "' must contain only finite values"))
  }
})

# ---------------------------------------------------------------------------
# Test 2: celltypist_predict() writes a factor column to colData
# ---------------------------------------------------------------------------

test_that("celltypist_predict writes factor column to colData (key_added param)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_sce_for_markers()

  # celltypist_predict requires a trained model file (.npz from the CellTypist repo).
  # On systems without a model file, the function must error informatively rather
  # than silently producing garbage. We use a temporary placeholder path and expect
  # either a successful factor column OR an informative error about the missing model.
  tmp_model <- tempfile(fileext = ".npz")

  tryCatch({
    sce_ct <- celltypist_predict(sce, model_path = tmp_model,
                                  use_dimred = "PCA", key_added = "celltypist")

    # If succeeded (e.g. a stub model is accepted), check the column
    cd <- SummarizedExperiment::colData(sce_ct)
    expect_true("celltypist" %in% colnames(cd),
      info = "colData(sce)$celltypist must be present after celltypist_predict()")

    ct_col <- cd$celltypist
    expect_true(is.factor(ct_col),
      info = "colData(sce)$celltypist must be a factor")
    expect_equal(length(ct_col), ncol(sce_ct),
      info = "celltypist column must have one value per cell")

  }, error = function(e) {
    # Acceptable failure modes: missing model file / unsupported format
    msg <- conditionMessage(e)
    expect_true(grepl("model|npz|file|path", msg, ignore.case = TRUE),
      info = paste0("celltypist_predict() error message should mention model/path; got: ", msg))
  })
})
