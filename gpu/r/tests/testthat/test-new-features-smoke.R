# test-new-features-smoke.R
# Cycle 52c: one test_that() block per feature (23 total).
# Parallel structure to python/tests/test_new_features_smoke.py.
#
# Each block:
#   1.  skip_if_not(singletgpu:::gpu_available())
#   2.  Build a tiny synthetic SCE / input (≤200 cells × 100 genes).
#   3.  Call the wrapper with seed=0L, resource="auto".
#   4.  Check named-list result / S4 class.
#
# Build: SKIPPED when GPU absent (singletgpu:::gpu_available() is FALSE).
# Gate : pending GPU dispatch.

library(testthat)

kSeed52c      <- 0xC0FFEEL
kNCells       <- 200L
kNGenes       <- 100L
kNSpots       <- 80L
kNRefCells    <- 60L
kNTypes       <- 4L
kNSnps        <- 50L
kNAdt         <- 20L
kNMotifs      <- 15L
kNPeaks       <- 80L
kNPcs         <- 20L
kNPerts       <- 5L

# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

.make_tiny_sce_smoke <- function(n_cells  = kNCells,
                                  n_genes  = kNGenes,
                                  seed     = kSeed52c,
                                  assay_nm = "counts") {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")
  set.seed(seed)
  m <- Matrix::Matrix(
    matrix(rnbinom(n_genes * n_cells, mu = 3, size = 1),
           nrow = n_genes, ncol = n_cells),
    sparse = TRUE
  )
  rownames(m) <- paste0("Gene", seq_len(n_genes))
  colnames(m) <- paste0("Cell", seq_len(n_cells))
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = m)
  )
  if (assay_nm != "counts") {
    SummarizedExperiment::assay(sce, assay_nm) <- m
  }
  sce
}

.add_logcounts <- function(sce, assay_src = "counts") {
  skip_if_not_installed("SummarizedExperiment")
  lc <- log1p(SummarizedExperiment::assay(sce, assay_src))
  SummarizedExperiment::assay(sce, "logcounts") <- lc
  sce
}

.add_pca <- function(sce, n_pcs = kNPcs, seed = kSeed52c) {
  set.seed(seed)
  n_cells <- ncol(sce)
  pca <- matrix(rnorm(n_cells * n_pcs), nrow = n_cells, ncol = n_pcs)
  rownames(pca) <- colnames(sce)
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca
  sce
}

.add_spatial <- function(sce, seed = kSeed52c) {
  set.seed(seed + 10L)
  n_spots <- ncol(sce)
  coords  <- matrix(runif(n_spots * 2L) * 1000, nrow = n_spots, ncol = 2L)
  colnames(coords) <- c("x", "y")
  rownames(coords) <- colnames(sce)
  SummarizedExperiment::colData(sce)$spatial <- coords
  sce
}

.make_sparse_matrix <- function(n_rows, n_cols, seed = kSeed52c) {
  skip_if_not_installed("Matrix")
  set.seed(seed)
  Matrix::Matrix(
    matrix(rnbinom(n_rows * n_cols, mu = 2, size = 1),
           nrow = n_rows, ncol = n_cols),
    sparse = TRUE
  )
}

.make_gene_net <- function(n_genes = kNGenes, n_sets = 5L, seed = kSeed52c) {
  set.seed(seed)
  data.frame(
    source = rep(paste0("Set", seq_len(n_sets)), each = 8L),
    target = paste0("Gene", sample.int(n_genes, n_sets * 8L, replace = TRUE)),
    stringsAsFactors = FALSE
  )
}

# ---------------------------------------------------------------------------
# Tier 1: cospar
# ---------------------------------------------------------------------------

test_that("cospar smoke: run_cospar returns SCE with obsm fate_bias", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  sce <- .add_logcounts(sce)
  sce <- .add_pca(sce)

  set.seed(kSeed52c)
  n_cells <- ncol(sce)
  SummarizedExperiment::colData(sce)$clone_id    <- sample(-1L:4L, n_cells, replace = TRUE)
  SummarizedExperiment::colData(sce)$time        <- sample(0L:2L,  n_cells, replace = TRUE)

  result <- run_cospar(sce,
                       clone_key     = "clone_id",
                       time_key      = "time",
                       basis         = "PCA",
                       k_neighbors   = 10L,
                       max_iters     = 5L,
                       n_fate_bias_steps = 2L,
                       seed          = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  expect_true("fate_bias" %in% names(SingleCellExperiment::reducedDims(result)),
    info = "run_cospar must write reducedDim 'fate_bias'")
  fb <- SingleCellExperiment::reducedDim(result, "fate_bias")
  expect_equal(nrow(fb), ncol(sce),
    info = paste0("fate_bias nrow ", nrow(fb), " != n_cells ", ncol(sce)))
})


# ---------------------------------------------------------------------------
# Tier 1: cellrank2
# ---------------------------------------------------------------------------

test_that("cellrank2 smoke: run_cellrank2 writes X_absorption_prob to reducedDim", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce <- .make_tiny_sce_smoke()
  n_cells <- ncol(sce)

  # Build a tiny row-stochastic transition matrix
  set.seed(kSeed52c)
  raw  <- matrix(runif(n_cells * n_cells), n_cells, n_cells)
  raw  <- raw / rowSums(raw)
  T_fwd <- Matrix::Matrix(raw, sparse = TRUE)
  SummarizedExperiment::metadata(sce)$T_fwd <- T_fwd

  terminal_cells <- colnames(sce)[1:5]
  SummarizedExperiment::colData(sce)$terminal_states <- ifelse(
    colnames(sce) %in% terminal_cells, "term", NA_character_
  )

  result <- run_cellrank2(sce,
                          transition_matrix_key = "T_fwd",
                          terminal_states_key   = "terminal_states",
                          gmres_m               = 10L,
                          gmres_max_restarts    = 3L,
                          seed                  = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  expect_true("X_absorption_prob" %in% names(SingleCellExperiment::reducedDims(result)),
    info = "run_cellrank2 must write reducedDim 'X_absorption_prob'")
  ap <- SingleCellExperiment::reducedDim(result, "X_absorption_prob")
  expect_equal(nrow(ap), n_cells,
    info = paste0("absorption_prob nrow ", nrow(ap), " != n_cells ", n_cells))
  expect_equal(ncol(ap), length(terminal_cells),
    info = paste0("absorption_prob ncol ", ncol(ap), " != n_terminals ", length(terminal_cells)))
})


# ---------------------------------------------------------------------------
# Tier 1: palantir
# ---------------------------------------------------------------------------

test_that("palantir smoke: run_palantir writes pseudotime and entropy to colData", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  sce <- .add_pca(sce)

  result <- run_palantir(sce,
                         start_cell      = colnames(sce)[1L],
                         terminal_states = colnames(sce)[c(2L, 10L, 20L)],
                         basis           = "PCA",
                         k_neighbors     = 10L,
                         seed            = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  cd_names <- names(SummarizedExperiment::colData(result))
  expect_true("pseudotime" %in% cd_names,
    info = "run_palantir must write colData$pseudotime")
  expect_true("entropy" %in% cd_names,
    info = "run_palantir must write colData$entropy")
  expect_equal(length(SummarizedExperiment::colData(result)$pseudotime), ncol(sce),
    info = "pseudotime length must equal n_cells")
})


# ---------------------------------------------------------------------------
# Tier 1: cellchat
# ---------------------------------------------------------------------------

test_that("cellchat smoke: run_cellchat writes metadata interaction_prob", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  set.seed(kSeed52c)
  SummarizedExperiment::colData(sce)$cell_type <- sample(
    paste0("Type", seq_len(kNTypes)), ncol(sce), replace = TRUE
  )
  # Tiny LR pairs
  n_lr <- 10L
  lr_db <- data.frame(
    ligand   = paste0("Gene", sample.int(kNGenes, n_lr, replace = TRUE)),
    receptor = paste0("Gene", sample.int(kNGenes, n_lr, replace = TRUE)),
    stringsAsFactors = FALSE
  )

  result <- run_cellchat(sce,
                         cell_type_key = "cell_type",
                         lr_database   = lr_db,
                         layer         = "counts",
                         seed          = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  meta <- SummarizedExperiment::metadata(result)
  expect_true("cellchat" %in% names(meta),
    info = "run_cellchat must write metadata$cellchat")
  expect_true("interaction_prob" %in% names(meta$cellchat),
    info = "cellchat metadata must contain 'interaction_prob'")
  iprob <- meta$cellchat$interaction_prob
  expect_equal(dim(iprob)[1L], kNTypes,
    info = paste0("interaction_prob dim[1] ", dim(iprob)[1L], " != n_types ", kNTypes))
})


# ---------------------------------------------------------------------------
# Tier 1: hdwgcna
# ---------------------------------------------------------------------------

test_that("hdwgcna smoke: run_hdwgcna writes metadata module_eigengenes", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  sce <- .add_logcounts(sce)
  sce <- .add_pca(sce)

  result <- run_hdwgcna(sce,
                        layer         = "logcounts",
                        k_neighbors   = 10L,
                        n_soft_power  = 5L,
                        resource      = "auto",
                        seed          = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  meta <- SummarizedExperiment::metadata(result)
  expect_true("hdwgcna" %in% names(meta),
    info = "run_hdwgcna must write metadata$hdwgcna")
  expect_true("module_eigengenes" %in% names(meta$hdwgcna),
    info = "hdwgcna metadata must contain 'module_eigengenes'")
  me <- meta$hdwgcna$module_eigengenes
  expect_equal(nrow(me), ncol(sce),
    info = paste0("module_eigengenes nrow ", nrow(me), " != n_cells ", ncol(sce)))
})


# ---------------------------------------------------------------------------
# Tier 1: milo
# ---------------------------------------------------------------------------

test_that("milo smoke: run_milo writes metadata nhoods with lfc and fdr", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  sce <- .add_pca(sce)
  set.seed(kSeed52c)
  SummarizedExperiment::colData(sce)$condition <- sample(
    c("ctrl", "case"), ncol(sce), replace = TRUE
  )

  result <- run_milo(sce,
                     condition_key = "condition",
                     ref_level     = "ctrl",
                     basis         = "PCA",
                     k_neighbors   = 10L,
                     seed          = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  meta <- SummarizedExperiment::metadata(result)
  expect_true("milo" %in% names(meta),
    info = "run_milo must write metadata$milo")
  expect_true("nhoods_lfc" %in% names(meta$milo),
    info = "milo metadata must contain 'nhoods_lfc'")
  expect_true("nhoods_fdr" %in% names(meta$milo),
    info = "milo metadata must contain 'nhoods_fdr'")
  expect_true(length(meta$milo$nhoods_lfc) > 0L,
    info = "milo nhoods_lfc must be non-empty")
})


# ---------------------------------------------------------------------------
# Tier 1: scdrs
# ---------------------------------------------------------------------------

test_that("scdrs smoke: run_scdrs writes colData cell_score", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  n_disease_genes <- 20L
  set.seed(kSeed52c)
  gene_weights <- data.frame(
    gene   = paste0("Gene", sample.int(kNGenes, n_disease_genes)),
    weight = runif(n_disease_genes),
    stringsAsFactors = FALSE
  )

  result <- run_scdrs(sce,
                      gene_weights  = gene_weights,
                      n_ctrl        = 5L,
                      layer         = "counts",
                      seed          = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  expect_true("cell_score" %in% names(SummarizedExperiment::colData(result)),
    info = "run_scdrs must write colData$cell_score")
  scores <- SummarizedExperiment::colData(result)$cell_score
  expect_equal(length(scores), ncol(sce),
    info = paste0("cell_score length ", length(scores), " != n_cells ", ncol(sce)))
  expect_true(all(is.finite(scores)),
    info = "cell_score must contain only finite values")
})


# ---------------------------------------------------------------------------
# Tier 2: granie
# ---------------------------------------------------------------------------

test_that("granie smoke: granie writes metadata gene_peak_links", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce      <- .make_tiny_sce_smoke()
  peak_mat <- .make_sparse_matrix(kNPeaks, kNCells, seed = kSeed52c + 1L)
  peak_sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = peak_mat)
  )
  rownames(peak_sce) <- paste0("Peak", seq_len(kNPeaks))
  colnames(peak_sce) <- colnames(sce)

  result <- granie(sce, peak_sce, layer = "counts", seed = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  meta <- SummarizedExperiment::metadata(result)
  expect_true("granie" %in% names(meta),
    info = "granie must write metadata$granie")
  expect_true("gene_peak_links" %in% names(meta$granie),
    info = "granie metadata must contain 'gene_peak_links'")
  links <- meta$granie$gene_peak_links
  expect_true(is.data.frame(links) || is.matrix(links),
    info = "gene_peak_links must be a data.frame or matrix")
})


# ---------------------------------------------------------------------------
# Tier 2: nebula
# ---------------------------------------------------------------------------

test_that("nebula smoke: run_eqtl returns list with beta, se, p_value", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce   <- .make_tiny_sce_smoke()
  snp_ad <- .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 2L)
  snp_dp <- snp_ad + .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 3L)

  set.seed(kSeed52c)
  n_donors <- 5L
  SummarizedExperiment::colData(sce)$donor_id <- sample.int(n_donors, kNCells, replace = TRUE)

  result <- run_eqtl(sce,
                     snp_ad    = snp_ad,
                     snp_dp    = snp_dp,
                     donor_key = "donor_id",
                     layer     = "counts",
                     seed      = 0L)

  for (key in c("beta", "se", "p_value")) {
    expect_true(key %in% names(result),
      info = paste0("run_eqtl result missing '", key, "'"))
  }
  expect_true("n_snps"  %in% names(result), info = "run_eqtl result missing 'n_snps'")
  expect_true("n_genes" %in% names(result), info = "run_eqtl result missing 'n_genes'")
})


# ---------------------------------------------------------------------------
# Tier 2: daesc
# ---------------------------------------------------------------------------

test_that("daesc smoke: run_ase returns list with beta, phi, p_value", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce   <- .make_tiny_sce_smoke()
  snp_ad <- .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 4L)
  snp_dp <- snp_ad + .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 5L)

  set.seed(kSeed52c)
  SummarizedExperiment::colData(sce)$cell_type <- sample(
    paste0("Type", seq_len(3L)), kNCells, replace = TRUE
  )

  result <- run_ase(sce,
                    snp_ad       = snp_ad,
                    snp_dp       = snp_dp,
                    cell_type_key = "cell_type",
                    seed         = 0L)

  for (key in c("beta", "phi", "p_value")) {
    expect_true(key %in% names(result),
      info = paste0("run_ase result missing '", key, "'"))
  }
  expect_equal(result$n_snps, kNSnps,
    info = paste0("run_ase n_snps ", result$n_snps, " != kNSnps ", kNSnps))
})


# ---------------------------------------------------------------------------
# Tier 2: numbat
# ---------------------------------------------------------------------------

test_that("numbat smoke: detect_cna returns list with cna_states and clone_labels", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce    <- .make_tiny_sce_smoke()
  snp_ad <- .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 6L)
  snp_dp <- snp_ad + .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 7L)

  set.seed(kSeed52c)
  chrom_labels <- sample.int(3L, kNSnps, replace = TRUE)

  result <- detect_cna(sce,
                       snp_ad       = snp_ad,
                       snp_dp       = snp_dp,
                       chrom_labels = chrom_labels,
                       layer        = "counts",
                       seed         = 0L)

  expect_true("cna_states"   %in% names(result),
    info = "detect_cna result missing 'cna_states'")
  expect_true("clone_labels" %in% names(result),
    info = "detect_cna result missing 'clone_labels'")
  expect_true("n_clones"     %in% names(result),
    info = "detect_cna result missing 'n_clones'")
})


# ---------------------------------------------------------------------------
# Tier 2: monopogen
# ---------------------------------------------------------------------------

test_that("monopogen smoke: call_variants returns list with germline_calls", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce    <- .make_tiny_sce_smoke()
  snp_ad <- .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 8L)
  snp_dp <- snp_ad + .make_sparse_matrix(kNSnps, kNCells, seed = kSeed52c + 9L)

  set.seed(kSeed52c)
  chrom_labels <- sample.int(3L, kNSnps, replace = TRUE)

  result <- call_variants(sce,
                          snp_ad       = snp_ad,
                          snp_dp       = snp_dp,
                          chrom_labels = chrom_labels,
                          seed         = 0L)

  expect_true("germline_calls"     %in% names(result),
    info = "call_variants result missing 'germline_calls'")
  expect_true("germline_genotypes" %in% names(result),
    info = "call_variants result missing 'germline_genotypes'")
  expect_equal(result$n_snps, kNSnps,
    info = paste0("call_variants n_snps ", result$n_snps, " != kNSnps ", kNSnps))
})


# ---------------------------------------------------------------------------
# Tier 2: chromvar
# ---------------------------------------------------------------------------

test_that("chromvar smoke: run_chromvar writes metadata deviation and variability", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  frag_mat <- .make_sparse_matrix(kNPeaks, kNCells, seed = kSeed52c + 10L)
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = frag_mat)
  )
  rownames(sce) <- paste0("Peak", seq_len(kNPeaks))
  colnames(sce) <- paste0("Cell", seq_len(kNCells))

  set.seed(kSeed52c + 11L)
  motif_in_peak <- Matrix::Matrix(
    matrix(sample(0L:1L, kNMotifs * kNPeaks, replace = TRUE),
           nrow = kNMotifs, ncol = kNPeaks),
    sparse = TRUE
  )
  rownames(motif_in_peak) <- paste0("Motif", seq_len(kNMotifs))

  result <- run_chromvar(sce,
                         motif_in_peak   = motif_in_peak,
                         n_background    = 5L,
                         layer           = "counts",
                         resource        = "auto",
                         seed            = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  meta <- SummarizedExperiment::metadata(result)
  expect_true("chromvar" %in% names(meta),
    info = "run_chromvar must write metadata$chromvar")
  expect_true("deviation"   %in% names(meta$chromvar),
    info = "chromvar metadata must contain 'deviation'")
  expect_true("variability" %in% names(meta$chromvar),
    info = "chromvar metadata must contain 'variability'")
  expect_equal(meta$chromvar$n_motifs, kNMotifs,
    info = paste0("chromvar n_motifs ", meta$chromvar$n_motifs, " != kNMotifs ", kNMotifs))
})


# ---------------------------------------------------------------------------
# Tier 3: flash_deconv
# ---------------------------------------------------------------------------

test_that("flash_deconv smoke: run_flash_deconv writes colData flash_deconv_props", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  spatial_sce <- .make_tiny_sce_smoke(n_cells = kNSpots)
  spatial_sce <- .add_spatial(spatial_sce)
  ref_sce     <- .make_tiny_sce_smoke(n_cells = kNRefCells)

  set.seed(kSeed52c)
  SummarizedExperiment::colData(ref_sce)$cell_type <- sample(
    paste0("Type", seq_len(kNTypes)), kNRefCells, replace = TRUE
  )

  result <- run_flash_deconv(spatial_sce,
                             ref_sce    = ref_sce,
                             ref_labels = "cell_type",
                             coords_key = "spatial",
                             layer      = "counts",
                             n_iter     = 5L,
                             tol        = 1e-3,
                             seed       = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  cd_names <- names(SummarizedExperiment::colData(result))
  expect_true("flash_deconv_props" %in% cd_names,
    info = "run_flash_deconv must write colData$flash_deconv_props")
  expect_true("flash_deconv_converged" %in% cd_names,
    info = "run_flash_deconv must write colData$flash_deconv_converged")
  props <- SummarizedExperiment::colData(result)$flash_deconv_props
  expect_equal(nrow(props), kNSpots,
    info = paste0("flash_deconv_props nrow ", nrow(props), " != kNSpots ", kNSpots))
  expect_equal(ncol(props), kNTypes,
    info = paste0("flash_deconv_props ncol ", ncol(props), " != kNTypes ", kNTypes))
})


# ---------------------------------------------------------------------------
# Tier 3: stagate (fit + predict)
# ---------------------------------------------------------------------------

test_that("stagate smoke: fit_stagate returns StagateModel; predict writes reducedDim", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke(n_cells = kNSpots)
  sce <- .add_logcounts(sce)
  sce <- .add_spatial(sce)

  model <- fit_stagate(sce,
                       coords_key = "spatial",
                       layer      = "logcounts",
                       n_latent   = 8L,
                       n_heads    = 2L,
                       n_epochs   = 3L,
                       resource   = "auto",
                       seed       = 0L)

  expect_s4_class(model, "StagateModel")
  expect_equal(model@n_latent, 8L)

  sce_pred <- methods::predict(model, newdata = sce, layer = "logcounts")
  expect_s4_class(sce_pred, "SingleCellExperiment")
  expect_true("STAGATE" %in% names(SingleCellExperiment::reducedDims(sce_pred)),
    info = "predict,StagateModel must write reducedDim 'STAGATE'")
  emb <- SingleCellExperiment::reducedDim(sce_pred, "STAGATE")
  expect_equal(nrow(emb), kNSpots,
    info = paste0("STAGATE embedding nrow ", nrow(emb), " != kNSpots ", kNSpots))
  expect_equal(ncol(emb), 8L,
    info = paste0("STAGATE embedding ncol ", ncol(emb), " != n_latent 8"))
})


# ---------------------------------------------------------------------------
# Tier 3: cell2fate (fit + predict)
# ---------------------------------------------------------------------------

test_that("cell2fate smoke: fit_cell2fate returns Cell2fateModel; predict writes cell2fate_latent_time", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce <- .make_tiny_sce_smoke()
  # Add spliced and unspliced assays
  spliced   <- SummarizedExperiment::assay(sce, "counts")
  unspliced <- Matrix::Matrix(
    matrix(pmax(0L, round(as.matrix(spliced) * 0.3)),
           nrow = nrow(spliced), ncol = ncol(spliced)),
    sparse = TRUE
  )
  dimnames(unspliced) <- dimnames(spliced)
  SummarizedExperiment::assay(sce, "spliced")   <- spliced
  SummarizedExperiment::assay(sce, "unspliced") <- unspliced

  model <- fit_cell2fate(sce,
                         layer_spliced   = "spliced",
                         layer_unspliced = "unspliced",
                         n_latent        = 8L,
                         n_epochs        = 3L,
                         resource        = "auto",
                         seed            = 0L)

  expect_s4_class(model, "Cell2fateModel")
  expect_equal(model@n_latent, 8L)

  sce_pred <- methods::predict(model, newdata = sce,
                               layer_spliced   = "spliced",
                               layer_unspliced = "unspliced")
  expect_s4_class(sce_pred, "SingleCellExperiment")
  expect_true("cell2fate_latent_time" %in% names(SummarizedExperiment::colData(sce_pred)),
    info = "predict,Cell2fateModel must write colData$cell2fate_latent_time")
  lt <- SummarizedExperiment::colData(sce_pred)$cell2fate_latent_time
  expect_equal(length(lt), kNCells,
    info = paste0("cell2fate_latent_time length ", length(lt), " != kNCells ", kNCells))
})


# ---------------------------------------------------------------------------
# Tier 3: discrete_diffusion (train + sample)
# ---------------------------------------------------------------------------

test_that("discrete_diffusion smoke: train returns DiscreteDiffusionModel; sample returns matrix", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()

  model <- train_discrete_diffusion(sce,
                                    layer    = "counts",
                                    n_latent = 8L,
                                    n_steps  = 5L,
                                    n_epochs = 2L,
                                    resource = "auto",
                                    seed     = 0L)

  expect_s4_class(model, "DiscreteDiffusionModel")
  expect_equal(model@n_latent, 8L)

  synth <- singletgpu::sample(model, n = 10L, seed = 0L)
  expect_true(is.matrix(synth) || methods::is(synth, "dgCMatrix"),
    info = "sample,DiscreteDiffusionModel must return a matrix or dgCMatrix")
  expect_equal(nrow(synth), kNGenes,
    info = paste0("sampled matrix nrow ", nrow(synth), " != kNGenes ", kNGenes))
  expect_equal(ncol(synth), 10L,
    info = "sampled matrix must have exactly 10 columns")
})


# ---------------------------------------------------------------------------
# Tier 3: perturb_graph (fit + predict)
# ---------------------------------------------------------------------------

test_that("perturb_graph smoke: fit returns PerturbGraphModel; predict writes metadata", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  sce <- .add_logcounts(sce)

  set.seed(kSeed52c)
  pert_labels <- c("control",
                   sample(paste0("Pert", seq_len(kNPerts)), kNCells - 1L, replace = TRUE))
  SummarizedExperiment::colData(sce)$perturbation <- pert_labels

  model <- fit_perturb_graph(sce,
                             perturb_key   = "perturbation",
                             control_label = "control",
                             layer         = "logcounts",
                             n_latent      = 8L,
                             n_epochs      = 3L,
                             resource      = "auto",
                             seed          = 0L)

  expect_s4_class(model, "PerturbGraphModel")
  expect_equal(model@n_latent, 8L)

  # Predict: use control cells only
  ctrl_idx <- which(SummarizedExperiment::colData(sce)$perturbation == "control")
  ctrl_sce <- sce[, ctrl_idx, drop = FALSE]
  sce_pred <- methods::predict(model, newdata = ctrl_sce, layer = "logcounts")
  expect_s4_class(sce_pred, "SingleCellExperiment")
  meta <- SummarizedExperiment::metadata(sce_pred)
  expect_true("perturb_predictions" %in% names(meta),
    info = "predict,PerturbGraphModel must write metadata$perturb_predictions")
  pp <- meta$perturb_predictions
  expect_true(length(pp) > 0L, info = "perturb_predictions must be non-empty")
  first <- pp[[1L]]
  expect_true("predicted" %in% names(first),
    info = "perturb_predictions[[1]] must contain 'predicted'")
  expect_true("delta"     %in% names(first),
    info = "perturb_predictions[[1]] must contain 'delta'")
})


# ---------------------------------------------------------------------------
# Tier 3: ssgsea
# ---------------------------------------------------------------------------

test_that("ssgsea smoke: run_ssgsea writes reducedDim ssgsea_scores", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce     <- .make_tiny_sce_smoke()
  n_sets  <- 5L
  net     <- .make_gene_net(n_genes = kNGenes, n_sets = n_sets)

  result  <- run_ssgsea(sce, net,
                        source    = "source",
                        target    = "target",
                        layer     = "counts",
                        normalize = TRUE,
                        seed      = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  expect_true("ssgsea_scores" %in% names(SingleCellExperiment::reducedDims(result)),
    info = "run_ssgsea must write reducedDim 'ssgsea_scores'")
  scores <- SingleCellExperiment::reducedDim(result, "ssgsea_scores")
  expect_equal(nrow(scores), kNCells,
    info = paste0("ssgsea_scores nrow ", nrow(scores), " != kNCells ", kNCells))
  expect_equal(ncol(scores), n_sets,
    info = paste0("ssgsea_scores ncol ", ncol(scores), " != n_sets ", n_sets))
})


# ---------------------------------------------------------------------------
# Tier 3: progeny
# ---------------------------------------------------------------------------

test_that("progeny smoke: run_progeny writes reducedDim progeny_scores", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce        <- .make_tiny_sce_smoke()
  n_pathways <- 4L
  set.seed(kSeed52c)
  pathway_weights <- data.frame(
    pathway = rep(paste0("Path", seq_len(n_pathways)), each = 10L),
    gene    = paste0("Gene", sample.int(kNGenes, n_pathways * 10L, replace = TRUE)),
    weight  = rnorm(n_pathways * 10L),
    stringsAsFactors = FALSE
  )

  result <- run_progeny(sce,
                        pathway_weights = pathway_weights,
                        layer           = "counts",
                        seed            = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  expect_true("progeny_scores" %in% names(SingleCellExperiment::reducedDims(result)),
    info = "run_progeny must write reducedDim 'progeny_scores'")
  scores <- SingleCellExperiment::reducedDim(result, "progeny_scores")
  expect_equal(nrow(scores), kNCells,
    info = paste0("progeny_scores nrow ", nrow(scores), " != kNCells ", kNCells))
  expect_equal(ncol(scores), n_pathways,
    info = paste0("progeny_scores ncol ", ncol(scores), " != n_pathways ", n_pathways))
})


# ---------------------------------------------------------------------------
# Tier 3: omnidoublet
# ---------------------------------------------------------------------------

test_that("omnidoublet smoke: detect_doublets writes colData omni_doublet_score", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")

  sce     <- .make_tiny_sce_smoke()
  adt_mat <- .make_sparse_matrix(kNAdt, kNCells, seed = kSeed52c + 20L)
  rownames(adt_mat) <- paste0("ADT", seq_len(kNAdt))
  colnames(adt_mat) <- colnames(sce)

  result <- detect_doublets(sce,
                            adt          = adt_mat,
                            n_pcs_rna    = 10L,
                            n_pcs_adt    = 5L,
                            n_hvg        = 50L,
                            k            = 10L,
                            layer        = "counts",
                            seed         = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  cd_names <- names(SummarizedExperiment::colData(result))
  expect_true("omni_doublet_score" %in% cd_names,
    info = "detect_doublets must write colData$omni_doublet_score")
  expect_true("omni_doublet_call"  %in% cd_names,
    info = "detect_doublets must write colData$omni_doublet_call")
  scores <- SummarizedExperiment::colData(result)$omni_doublet_score
  expect_true(all(is.finite(scores)),
    info = "omni_doublet_score must be finite")
})


# ---------------------------------------------------------------------------
# Tier 3: doublet_score
# ---------------------------------------------------------------------------

test_that("doublet_score smoke: scrublet_score writes colData doublet_score", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()
  sce <- .add_pca(sce)

  result <- scrublet_score(sce,
                           embedding_key     = "PCA",
                           n_synth_frac      = 0.25,
                           k                 = 10L,
                           obs_score_key     = "doublet_score",
                           obs_call_key      = "doublet_call",
                           seed              = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  cd_names <- names(SummarizedExperiment::colData(result))
  expect_true("doublet_score" %in% cd_names,
    info = "scrublet_score must write colData$doublet_score")
  expect_true("doublet_call"  %in% cd_names,
    info = "scrublet_score must write colData$doublet_call")
})


# ---------------------------------------------------------------------------
# Tier 3: csi_gep
# ---------------------------------------------------------------------------

test_that("csi_gep smoke: run_csi_gep writes rowData programs and colData usage", {
  skip_if_not(singletgpu:::gpu_available(), "GPU not available")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")

  sce <- .make_tiny_sce_smoke()

  result <- run_csi_gep(sce,
                        k_range       = c(3L, 5L),
                        n_runs        = 5L,
                        subsample_frac = 0.8,
                        nmf_max_iter  = 10L,
                        layer         = "counts",
                        resource      = "auto",
                        seed          = 0L)

  expect_s4_class(result, "SingleCellExperiment")
  rd <- SummarizedExperiment::rowData(result)
  expect_true("csi_gep_programs" %in% colnames(rd),
    info = "run_csi_gep must write rowData$csi_gep_programs")

  expect_true("csi_gep_usage" %in% names(SingleCellExperiment::reducedDims(result)),
    info = "run_csi_gep must write reducedDim 'csi_gep_usage'")

  k_chosen <- ncol(rd$csi_gep_programs)
  expect_true(k_chosen %in% c(3L, 5L),
    info = paste0("csi_gep chose k=", k_chosen, " not in c(3, 5)"))

  usage <- SingleCellExperiment::reducedDim(result, "csi_gep_usage")
  expect_equal(nrow(usage), kNCells,
    info = paste0("csi_gep_usage nrow ", nrow(usage), " != kNCells ", kNCells))
  expect_equal(ncol(usage), k_chosen,
    info = paste0("csi_gep_usage ncol ", ncol(usage), " != k_chosen ", k_chosen))
})
