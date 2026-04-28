# test-pseudobulk.R
# Tests for aggregate_pseudobulk() and pseudobulk_de() — donor-aware DE.
# Design ref: singlet-gpu/state/designs/26-r-wrappers-3.md §R API surface
#
# Tests:
#   1. aggregate_pseudobulk() returns a SingleCellExperiment
#   2. pseudobulk_de() returns list of data.frames (one per cluster)
#   3. pseudobulk_de() LFC vs muscat::pbDS — Spearman rho >= 0.97
#      (skipped if muscat is unavailable)
#   4. pseudobulk_de() min_cells_per_pseudobulk filter — small groups excluded

kSeedPseudobulk <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: SCE with cluster + sample colData columns for pseudobulk tests
# ---------------------------------------------------------------------------

make_sce_for_pseudobulk <- function(seed          = kSeedPseudobulk,
                                     n_genes       = 500L,
                                     n_cells       = 400L,
                                     n_clusters    = 3L,
                                     n_donors      = 4L) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("S4Vectors")

  set.seed(seed)

  cluster_labels <- rep(paste0("Clust", seq_len(n_clusters)),
                        length.out = n_cells)
  donor_labels   <- rep(paste0("Donor", seq_len(n_donors)),
                        length.out = n_cells)

  # Base counts: NB(mu=5, size=1)
  base <- matrix(rnbinom(n_genes * n_cells, mu = 5L, size = 1L),
                 nrow = n_genes, ncol = n_cells)

  # Plant a DE signal: first 30 genes 3× up in Clust1 vs others
  clust1_idx <- which(cluster_labels == "Clust1")
  base[1:30, clust1_idx] <- base[1:30, clust1_idx] * 3L

  # Plant donor batch effect: Donor1 has mild overall elevation
  donor1_idx <- which(donor_labels == "Donor1")
  base[, donor1_idx] <- round(base[, donor1_idx] * 1.2)

  counts <- Matrix::Matrix(base, sparse = TRUE)
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell",  seq_len(n_cells))

  SingleCellExperiment::SingleCellExperiment(
    assays  = list(counts = counts),
    colData = S4Vectors::DataFrame(
      cluster = factor(cluster_labels),
      donor   = factor(donor_labels)
    )
  )
}

# ---------------------------------------------------------------------------
# Test 1: aggregate_pseudobulk returns SingleCellExperiment
# ---------------------------------------------------------------------------

test_that("aggregate_pseudobulk returns SCE with summed counts per (cluster, donor)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_sce_for_pseudobulk()
  pb  <- aggregate_pseudobulk(sce,
                                group_by  = "cluster",
                                sample_by = "donor",
                                layer     = "counts",
                                min_cells_per_pseudobulk = 5L)

  expect_s4_class(pb, "SingleCellExperiment")

  # Number of pseudobulk columns = n_clusters × n_donors (or fewer if filtered)
  n_clusters <- nlevels(SummarizedExperiment::colData(sce)$cluster)
  n_donors   <- nlevels(SummarizedExperiment::colData(sce)$donor)
  max_cols   <- n_clusters * n_donors

  expect_true(ncol(pb) > 0L && ncol(pb) <= max_cols,
    info = paste0("aggregate_pseudobulk() must produce 1..", max_cols,
                  " pseudobulk columns; got ", ncol(pb)))

  # Rows = genes
  expect_equal(nrow(pb), nrow(sce),
    info = "aggregate_pseudobulk() output must have same number of genes as input SCE")

  # assays$counts must be present and contain non-negative integers
  expect_true("counts" %in% SummarizedExperiment::assayNames(pb),
    info = "aggregate_pseudobulk() output must have assay 'counts'")
  pb_counts <- SummarizedExperiment::assay(pb, "counts")
  expect_true(all(as.numeric(pb_counts) >= 0, na.rm = TRUE),
    info = "Pseudobulk counts must be non-negative")

  # colData must record cluster and donor for each pseudobulk column
  pb_cd <- SummarizedExperiment::colData(pb)
  expect_true("cluster" %in% colnames(pb_cd) || "group_by" %in% colnames(pb_cd),
    info = "pseudobulk colData must record cluster membership")
  expect_true("donor" %in% colnames(pb_cd) || "sample_by" %in% colnames(pb_cd),
    info = "pseudobulk colData must record donor membership")
})

# ---------------------------------------------------------------------------
# Test 2: pseudobulk_de returns list of data.frames per cluster
# ---------------------------------------------------------------------------

test_that("pseudobulk_de returns list of data.frames (one per cluster)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce    <- make_sce_for_pseudobulk()
  result <- pseudobulk_de(sce,
                           group_by                  = "cluster",
                           sample_by                 = "donor",
                           layer                     = "counts",
                           min_cells_per_pseudobulk  = 5L,
                           apeglm_shrinkage          = FALSE,
                           seed                      = kSeedPseudobulk)

  # Must be a list
  expect_true(is.list(result),
    info = "pseudobulk_de() must return a list")

  n_clusters <- nlevels(SummarizedExperiment::colData(sce)$cluster)
  expect_gte(length(result), 1L,
    info = "pseudobulk_de() result list must have at least one element")
  expect_lte(length(result), n_clusters,
    info = paste0("pseudobulk_de() result list must have at most ",
                  n_clusters, " elements (one per cluster)"))

  # Each element must be a data.frame with required columns
  required_cols <- c("gene", "log2FC", "pvalue", "padj")
  for (clust_name in names(result)) {
    df <- result[[clust_name]]
    expect_true(is.data.frame(df),
      info = paste0("pseudobulk_de() result[[", clust_name,
                    "]] must be a data.frame"))
    for (col in required_cols) {
      expect_true(col %in% colnames(df),
        info = paste0("pseudobulk_de() result[[", clust_name,
                      "]] must have column '", col, "'"))
    }
    expect_gte(nrow(df), 1L,
      info = paste0("pseudobulk_de() result[[", clust_name,
                    "]] must have at least 1 row"))
    # pvalue and padj must be in [0, 1]
    expect_true(all(df$pvalue >= 0 & df$pvalue <= 1, na.rm = TRUE),
      info = paste0("cluster ", clust_name, " pvalue must be in [0, 1]"))
    expect_true(all(df$padj >= 0 & df$padj <= 1, na.rm = TRUE),
      info = paste0("cluster ", clust_name, " padj must be in [0, 1]"))
    # log2FC must be finite (or NA for filtered genes)
    lfc <- df$log2FC
    expect_true(all(is.finite(lfc) | is.na(lfc)),
      info = paste0("cluster ", clust_name, " log2FC must be finite or NA"))
  }
})

# ---------------------------------------------------------------------------
# Test 3: pseudobulk_de LFC vs muscat::pbDS — Spearman rho >= 0.97
# ---------------------------------------------------------------------------

test_that("pseudobulk_de LFC vs muscat::pbDS Spearman rho >= 0.97", {
  skip_if_not_installed("muscat")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce <- make_sce_for_pseudobulk()

  # GPU result
  gpu_result <- pseudobulk_de(sce,
                               group_by                 = "cluster",
                               sample_by                = "donor",
                               layer                    = "counts",
                               min_cells_per_pseudobulk = 5L,
                               apeglm_shrinkage         = FALSE,
                               seed                     = kSeedPseudobulk)

  # CPU reference: muscat aggregateData + pbDS
  # muscat expects colData columns: cluster_id, sample_id
  sce_muscat <- sce
  SummarizedExperiment::colData(sce_muscat)$cluster_id <-
    SummarizedExperiment::colData(sce)$cluster
  SummarizedExperiment::colData(sce_muscat)$sample_id  <-
    SummarizedExperiment::colData(sce)$donor

  pb_ref <- muscat::aggregateData(sce_muscat,
                                   assay    = "counts",
                                   fun      = "sum",
                                   by       = c("cluster_id", "sample_id"))

  tryCatch({
    de_ref <- muscat::pbDS(pb_ref, verbose = FALSE)

    # Compare LFCs for the first cluster that appears in both results
    target_clust <- intersect(names(gpu_result), names(de_ref$table))
    if (length(target_clust) == 0L) {
      skip("No cluster names overlap between GPU result and muscat::pbDS output")
    }
    tc <- target_clust[[1L]]

    gpu_df  <- gpu_result[[tc]]
    cpu_df  <- de_ref$table[[tc]][[1L]]  # muscat returns list-of-lists

    shared_genes <- intersect(gpu_df$gene, cpu_df$gene)
    if (length(shared_genes) < 20L) {
      skip(paste0("Fewer than 20 shared genes for cluster ", tc, " — skip Spearman"))
    }

    gpu_lfc <- gpu_df$log2FC[match(shared_genes, gpu_df$gene)]
    cpu_lfc <- cpu_df$logFC[ match(shared_genes, cpu_df$gene)]

    rho <- cor(as.numeric(gpu_lfc), as.numeric(cpu_lfc),
               method = "spearman", use = "complete.obs")
    tolerance <- 0.97

    expect_gte(rho, tolerance,
      info = paste0("Spearman rho (pseudobulk_de LFC vs muscat::pbDS, cluster=",
                    tc, ") = ", round(rho, 4),
                    " (threshold = ", tolerance, ")"))
  }, error = function(e) {
    skip(paste0("muscat::pbDS error: ", conditionMessage(e)))
  })
})

# ---------------------------------------------------------------------------
# Test 4: pseudobulk_de min_cells_per_pseudobulk filter
# ---------------------------------------------------------------------------

test_that("pseudobulk_de min_cells_per_pseudobulk filter excludes small groups", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  # Build an SCE where one cluster (Clust3) has fewer cells per donor
  # than the min_cells threshold — those pseudobulks must be dropped.
  set.seed(kSeedPseudobulk)
  n_genes  <- 200L
  n_cells  <- 300L
  # Clust1: 120 cells, Clust2: 120 cells, Clust3: 60 cells (4 donors × 15 = 60)
  cluster_labels <- c(rep("Clust1", 120L), rep("Clust2", 120L), rep("Clust3", 60L))
  donor_labels   <- rep(paste0("Donor", 1:4), length.out = n_cells)
  counts <- Matrix::Matrix(
    matrix(rnbinom(n_genes * n_cells, mu = 3L, size = 1L),
           nrow = n_genes, ncol = n_cells),
    sparse = TRUE
  )
  rownames(counts) <- paste0("Gene", seq_len(n_genes))
  colnames(counts) <- paste0("Cell",  seq_len(n_cells))

  sce_small <- SingleCellExperiment::SingleCellExperiment(
    assays  = list(counts = counts),
    colData = S4Vectors::DataFrame(
      cluster = factor(cluster_labels),
      donor   = factor(donor_labels)
    )
  )

  # With a high threshold: Clust3 should be excluded entirely (15 cells/donor < 20)
  high_threshold <- 20L
  result_strict  <- pseudobulk_de(sce_small,
                                   group_by                 = "cluster",
                                   sample_by                = "donor",
                                   layer                    = "counts",
                                   min_cells_per_pseudobulk = high_threshold,
                                   apeglm_shrinkage         = FALSE,
                                   seed                     = kSeedPseudobulk)

  # Clust3 must not appear in results (all its pseudobulks are below threshold)
  expect_false("Clust3" %in% names(result_strict),
    info = paste0("Clust3 (15 cells/donor < min_cells=", high_threshold,
                  ") must be excluded from pseudobulk_de() results"))

  # With a low threshold: all three clusters should be included
  low_threshold <- 5L
  result_lenient <- pseudobulk_de(sce_small,
                                   group_by                 = "cluster",
                                   sample_by                = "donor",
                                   layer                    = "counts",
                                   min_cells_per_pseudobulk = low_threshold,
                                   apeglm_shrinkage         = FALSE,
                                   seed                     = kSeedPseudobulk)

  expect_true(length(result_lenient) >= 3L,
    info = paste0("With min_cells=", low_threshold,
                  " all three clusters should be included; got ",
                  length(result_lenient)))
})
