# test-lineage.R
# Tests for detect_clones() — MT heteroplasmy-based lineage tracing.
# Design ref: singlet-gpu/state/designs/26-r-wrappers-3.md §R API surface
#
# Tests:
#   1. detect_clones() writes colData$mt_clone_id
#   2. detect_clones() writes metadata$mt_lineage with informative_sites
#   3. detect_clones() real-data smoke on GSM4037629 (mt_heteroplasmy.1pz)
#   4. detect_clones() min_K constraint respected

kSeedLineage <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: SCE with mt_alt and mt_depth assays simulating heteroplasmy
# ---------------------------------------------------------------------------

make_sce_for_lineage <- function(seed = kSeedLineage,
                                  n_sites = 40L,
                                  n_cells = 200L,
                                  n_clones = 3L) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("S4Vectors")

  set.seed(seed)

  # Total depth at each MT site per cell: ~100–500 reads
  depth <- matrix(sample(100L:500L, n_sites * n_cells, replace = TRUE),
                  nrow = n_sites, ncol = n_cells)

  # Clone assignments (ground truth — used to plant heteroplasmy signals)
  clone_id <- rep(seq_len(n_clones), length.out = n_cells)

  # Alternate allele counts — each clone has a distinct set of high-VAF sites
  alt <- matrix(0L, nrow = n_sites, ncol = n_cells)
  sites_per_clone <- ceiling(n_sites / n_clones)
  for (cl in seq_len(n_clones)) {
    site_start <- (cl - 1L) * sites_per_clone + 1L
    site_end   <- min(cl * sites_per_clone, n_sites)
    cl_cells   <- which(clone_id == cl)
    alt[site_start:site_end, cl_cells] <- round(
      depth[site_start:site_end, cl_cells] *
        runif(length(site_start:site_end) * length(cl_cells), 0.6, 0.95)
    )
  }

  rownames(alt)   <- paste0("MT_site_", seq_len(n_sites))
  colnames(alt)   <- paste0("Cell_", seq_len(n_cells))
  rownames(depth) <- rownames(alt)
  colnames(depth) <- colnames(alt)

  alt_sp   <- Matrix::Matrix(alt,   sparse = TRUE)
  depth_sp <- Matrix::Matrix(depth, sparse = TRUE)

  SingleCellExperiment::SingleCellExperiment(
    assays  = list(mt_alt = alt_sp, mt_depth = depth_sp),
    colData = S4Vectors::DataFrame(true_clone = factor(clone_id))
  )
}

# ---------------------------------------------------------------------------
# Test 1: detect_clones writes colData$mt_clone_id
# ---------------------------------------------------------------------------

test_that("detect_clones writes colData$mt_clone_id", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce    <- make_sce_for_lineage()
  sce_cl <- detect_clones(sce,
                           alt_assay    = "mt_alt",
                           depth_assay  = "mt_depth",
                           min_depth    = 10L,
                           min_cells_alt = 3L,
                           min_vaf      = 0.01,
                           min_K        = 2L,
                           max_K        = 6L,
                           seed         = kSeedLineage)

  expect_s4_class(sce_cl, "SingleCellExperiment")

  cd <- SummarizedExperiment::colData(sce_cl)
  expect_true("mt_clone_id" %in% colnames(cd),
    info = paste0("detect_clones() must write colData$mt_clone_id; ",
                  "available cols: ", paste(colnames(cd), collapse = ", ")))

  clone_id_vals <- cd$mt_clone_id
  expect_equal(length(clone_id_vals), ncol(sce_cl),
    info = "mt_clone_id must have one value per cell")
  expect_true(is.factor(clone_id_vals) || is.integer(clone_id_vals) ||
                is.character(clone_id_vals),
    info = "mt_clone_id must be a factor, integer, or character vector")

  # There must be at least 2 distinct clone labels (min_K = 2)
  n_clones_detected <- length(unique(clone_id_vals))
  expect_gte(n_clones_detected, 2L,
    info = paste0("detect_clones() must assign at least min_K=2 clones; got ",
                  n_clones_detected))
})

# ---------------------------------------------------------------------------
# Test 2: detect_clones writes metadata$mt_lineage
# ---------------------------------------------------------------------------

test_that("detect_clones writes metadata$mt_lineage with informative_sites", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce    <- make_sce_for_lineage()
  sce_cl <- detect_clones(sce,
                           alt_assay     = "mt_alt",
                           depth_assay   = "mt_depth",
                           min_depth     = 10L,
                           min_cells_alt = 3L,
                           min_vaf       = 0.01,
                           min_K         = 2L,
                           max_K         = 6L,
                           seed          = kSeedLineage)

  lin_meta <- S4Vectors::metadata(sce_cl)$mt_lineage
  expect_true(!is.null(lin_meta),
    info = "detect_clones() must write metadata(sce)$mt_lineage")

  # Must contain informative_sites element
  expect_true("informative_sites" %in% names(lin_meta),
    info = paste0("metadata$mt_lineage must contain 'informative_sites'; ",
                  "got: ", paste(names(lin_meta), collapse = ", ")))

  inf_sites <- lin_meta$informative_sites
  expect_true(is.character(inf_sites) || is.integer(inf_sites),
    info = "mt_lineage$informative_sites must be a character or integer vector of site IDs")
  expect_gte(length(inf_sites), 1L,
    info = "mt_lineage$informative_sites must list at least one informative site")
})

# ---------------------------------------------------------------------------
# Test 3: real-data smoke on GSM4037629 (mt_heteroplasmy.1pz)
# ---------------------------------------------------------------------------

test_that("detect_clones GSM4037629 real-data smoke (mt_heteroplasmy.1pz)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gsm4037629()
  skip_if_no_gpu()

  gsm_dir <- gsm4037629_path()
  mt_path <- file.path(gsm_dir, "mt_heteroplasmy.1pz")
  if (!file.exists(mt_path)) {
    skip(paste0("mt_heteroplasmy.1pz not present at ", mt_path))
  }

  # Load MT heteroplasmy SCE using the package helper
  sce_mt <- read_pz_mt_sce(gsm_dir)
  expect_s4_class(sce_mt, "SingleCellExperiment")

  # Run clone detection with permissive thresholds
  sce_cl <- detect_clones(sce_mt,
                           alt_assay     = "mt_alt",
                           depth_assay   = "mt_depth",
                           min_depth     = 10L,
                           min_cells_alt = 5L,
                           min_vaf       = 0.01,
                           min_K         = 2L,
                           max_K         = 8L,
                           seed          = kSeedLineage)

  expect_s4_class(sce_cl, "SingleCellExperiment")
  expect_true("mt_clone_id" %in%
                colnames(SummarizedExperiment::colData(sce_cl)),
    info = "GSM4037629 smoke: mt_clone_id not written to colData")

  # Cell count must match GSM4037629 expected
  expected_cells <- gsm4037629_expected_cells()
  expect_equal(ncol(sce_cl), expected_cells,
    info = paste0("Expected ", expected_cells, " cells from GSM4037629; got ",
                  ncol(sce_cl)))
})

# ---------------------------------------------------------------------------
# Test 4: detect_clones min_K constraint
# ---------------------------------------------------------------------------

test_that("detect_clones min_K constraint — result has >= min_K clone levels", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  for (min_K_val in c(2L, 3L)) {
    sce    <- make_sce_for_lineage(n_clones = 4L)
    sce_cl <- detect_clones(sce,
                             alt_assay     = "mt_alt",
                             depth_assay   = "mt_depth",
                             min_depth     = 10L,
                             min_cells_alt = 3L,
                             min_vaf       = 0.01,
                             min_K         = min_K_val,
                             max_K         = 8L,
                             seed          = kSeedLineage)

    clone_vals <- SummarizedExperiment::colData(sce_cl)$mt_clone_id
    n_detected <- length(unique(clone_vals))

    expect_gte(n_detected, min_K_val,
      info = paste0("detect_clones() with min_K=", min_K_val,
                    " must produce >= ", min_K_val,
                    " distinct clone labels; got ", n_detected))
  }
})
