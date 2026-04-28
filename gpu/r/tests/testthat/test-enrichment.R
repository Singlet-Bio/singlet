# test-enrichment.R
# Tests for run_gsea() and run_aucell() — GPU gene-set enrichment functions.
# Design ref: singlet-gpu/state/designs/26-r-wrappers-3.md §R API surface
#
# Tests:
#   1. run_gsea() returns a data.frame with required columns
#   2. run_gsea() NES vs fgsea::fgsea — Spearman rho >= 0.95
#   3. run_aucell() writes one AUC column per set to colData(sce)
#   4. run_aucell() AUC scores vs AUCell::AUCell_run — Spearman rho >= 0.95

kSeedGSEA <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Named numeric rank-stat vector for GSEA tests.
#' 500 genes; first 50 inflate to create a real enrichment signal for PathA.
make_gsea_stats <- function(seed = kSeedGSEA) {
  set.seed(seed)
  genes <- paste0("Gene", seq_len(500L))
  stats <- rnorm(500L)
  # inflate signal for first 50 genes (PathA)
  stats[1:50] <- stats[1:50] + 2.5
  names(stats) <- genes
  sort(stats, decreasing = TRUE)  # fgsea expects a sorted named vector
}

#' Named list of gene sets for GSEA / AUCell tests.
make_gene_sets <- function() {
  list(
    PathA = paste0("Gene", 1:50),    # enriched (inflated above)
    PathB = paste0("Gene", 251:300), # neutral
    PathC = paste0("Gene", 451:500)  # depleted (tail)
  )
}

#' Minimal SCE with counts assay for AUCell tests.
make_sce_for_aucell <- function(seed = kSeedGSEA) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")
  set.seed(seed)
  n_genes <- 500L
  n_cells <- 200L
  dense <- matrix(rnbinom(n_genes * n_cells, mu = 3, size = 1),
                  nrow = n_genes, ncol = n_cells)
  # inflate PathA genes in first 60 cells
  dense[1:50, 1:60] <- dense[1:50, 1:60] * 4L
  counts <- Matrix::Matrix(dense, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell",  seq_len(n_cells))
  SingleCellExperiment::SingleCellExperiment(assays = list(counts = counts))
}

# ---------------------------------------------------------------------------
# Test 1: run_gsea returns data.frame with required columns
# ---------------------------------------------------------------------------

test_that("run_gsea returns data.frame with required columns", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_gpu()

  stats    <- make_gsea_stats()
  pathways <- make_gene_sets()

  result <- run_gsea(stats, pathways,
                     eps = 0.0, min_size = 5L, max_size = 500L,
                     seed = kSeedGSEA, n_perm = 100L)

  expect_true(is.data.frame(result),
    info = "run_gsea() must return a data.frame")

  required_cols <- c("pathway", "ES", "NES", "pval", "padj")
  for (col in required_cols) {
    expect_true(col %in% colnames(result),
      info = paste0("run_gsea() result must contain column '", col, "'"))
  }

  # One row per pathway
  expect_equal(nrow(result), length(pathways),
    info = paste0("Expected ", length(pathways), " rows (one per pathway); got ",
                  nrow(result)))

  # NES must be finite
  expect_true(all(is.finite(result$NES)),
    info = "run_gsea() NES values must all be finite")

  # pval must be in [0, 1]
  expect_true(all(result$pval >= 0 & result$pval <= 1),
    info = "run_gsea() pval must be in [0, 1]")

  # PathA must show positive NES (inflated signal)
  path_a_nes <- result$NES[result$pathway == "PathA"]
  expect_true(length(path_a_nes) == 1L && path_a_nes > 0,
    info = "PathA (inflated genes) should have positive NES")
})

# ---------------------------------------------------------------------------
# Test 2: run_gsea NES vs fgsea::fgsea — Spearman rho >= 0.95
# ---------------------------------------------------------------------------

test_that("run_gsea NES vs fgsea::fgsea Spearman rho >= 0.95", {
  skip_if_not_installed("fgsea")
  skip_if_no_gpu()

  stats    <- make_gsea_stats()
  pathways <- make_gene_sets()

  # GPU result
  gpu_res <- run_gsea(stats, pathways,
                      eps = 0.0, min_size = 5L, max_size = 500L,
                      seed = kSeedGSEA, n_perm = 1000L)

  # CPU reference: fgsea::fgsea
  cpu_res <- fgsea::fgsea(
    pathways = pathways,
    stats    = stats,
    eps      = 0.0,
    minSize  = 5L,
    maxSize  = 500L,
    nPermSimple = 1000L,
    seed    = kSeedGSEA
  )

  # Align on pathway name
  shared_paths <- intersect(gpu_res$pathway, cpu_res$pathway)
  if (length(shared_paths) < 2L) {
    skip("Fewer than 2 shared pathways between GPU and fgsea results")
  }

  gpu_nes <- gpu_res$NES[match(shared_paths, gpu_res$pathway)]
  cpu_nes <- cpu_res$NES[match(shared_paths, cpu_res$pathway)]

  rho <- cor(gpu_nes, cpu_nes, method = "spearman", use = "complete.obs")
  tolerance <- 0.95

  expect_gte(rho, tolerance,
    info = paste0("Spearman rho (run_gsea NES vs fgsea::fgsea) = ",
                  round(rho, 4), " (threshold = ", tolerance, ")"))
})

# ---------------------------------------------------------------------------
# Test 3: run_aucell writes colData per set
# ---------------------------------------------------------------------------

test_that("run_aucell writes colData per set (AUC_<set> columns)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce       <- make_sce_for_aucell()
  gene_sets <- make_gene_sets()

  sce_scored <- run_aucell(sce, gene_sets = gene_sets,
                            layer = "counts", normAUC = TRUE)

  expect_s4_class(sce_scored, "SingleCellExperiment")

  cd <- SummarizedExperiment::colData(sce_scored)

  for (set_name in names(gene_sets)) {
    expected_col <- paste0("AUC_", set_name)
    expect_true(expected_col %in% colnames(cd),
      info = paste0("colData must contain column '", expected_col,
                    "' after run_aucell()"))
    auc_vals <- as.numeric(cd[[expected_col]])
    expect_equal(length(auc_vals), ncol(sce_scored),
      info = paste0("'", expected_col, "' must have one value per cell"))
    expect_true(all(is.finite(auc_vals)),
      info = paste0("'", expected_col, "' must contain only finite values"))
    if (isTRUE(attr(sce_scored, "normAUC"))) {
      # normalized AUC should be in [0, 1]
      expect_true(all(auc_vals >= 0 & auc_vals <= 1 + .Machine$double.eps),
        info = paste0("normalized '", expected_col, "' must lie in [0, 1]"))
    }
  }
})

# ---------------------------------------------------------------------------
# Test 4: run_aucell AUC vs AUCell::AUCell_run — Spearman rho >= 0.95
# ---------------------------------------------------------------------------

test_that("run_aucell AUC vs AUCell::AUCell_run Spearman rho >= 0.95", {
  skip_if_not_installed("AUCell")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_no_gpu()

  sce       <- make_sce_for_aucell()
  gene_sets <- make_gene_sets()

  # GPU result
  sce_gpu <- run_aucell(sce, gene_sets = gene_sets,
                         layer = "counts", normAUC = TRUE)

  # CPU reference: AUCell::AUCell_run
  expr_mat <- SummarizedExperiment::assay(sce, "counts")
  cell_rank <- AUCell::AUCell_buildRankings(expr_mat, nCores = 1L, plotStats = FALSE)
  auc_ref   <- AUCell::AUCell_calcAUC(gene_sets, cell_rank, normAUC = TRUE,
                                       aucMaxRank = ceiling(0.05 * nrow(sce)))

  # Compare per-set AUC scores across cells
  rho_values <- vapply(names(gene_sets), function(set_name) {
    gpu_col <- paste0("AUC_", set_name)
    gpu_auc <- as.numeric(SummarizedExperiment::colData(sce_gpu)[[gpu_col]])
    cpu_auc <- as.numeric(AUCell::getAUC(auc_ref)[set_name, ])
    cor(gpu_auc, cpu_auc, method = "spearman", use = "complete.obs")
  }, numeric(1L))

  tolerance <- 0.95
  for (set_name in names(rho_values)) {
    expect_gte(rho_values[[set_name]], tolerance,
      info = paste0("Spearman rho (run_aucell vs AUCell, set=", set_name, ") = ",
                    round(rho_values[[set_name]], 4),
                    " (threshold = ", tolerance, ")"))
  }
})
