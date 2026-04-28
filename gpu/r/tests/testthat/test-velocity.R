# test-velocity.R
# Tests for velocity_moments() and velocity_prep() — RNA velocity prep functions.
# Design ref: singlet-gpu/state/designs/26-r-wrappers-3.md §R API surface
#
# Tests:
#   1. velocity_moments() writes assays Ms and Mu (smoothed moments)
#   2. velocity_prep() writes assays$velocity and rowData$velocity_gamma
#   3. velocity_prep() gamma vs velociraptor::scvelo — Spearman rho >= 0.95
#      (skipped if velociraptor is unavailable)

kSeedVelocity <- 0xC0FFEE

# ---------------------------------------------------------------------------
# Helper: SCE with spliced + unspliced assays and a PCA for kNN smoothing
# ---------------------------------------------------------------------------

make_sce_for_velocity <- function(seed = kSeedVelocity,
                                   n_genes = 300L,
                                   n_cells = 200L,
                                   n_pcs   = 20L) {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("S4Vectors")

  set.seed(seed)

  # Spliced counts: NB(mu=5, size=1) — simulates mature mRNA
  spliced <- Matrix::Matrix(
    matrix(rnbinom(n_genes * n_cells, mu = 5L, size = 1L),
           nrow = n_genes, ncol = n_cells),
    sparse = TRUE
  )
  rownames(spliced) <- paste0("Gene", seq_len(n_genes))
  colnames(spliced) <- paste0("Cell",  seq_len(n_cells))

  # Unspliced counts: correlated with spliced but lower magnitude
  unspliced <- Matrix::Matrix(
    matrix(pmax(0L, round(as.matrix(spliced) * 0.3 +
                           rnorm(n_genes * n_cells, 0, 0.5))),
           nrow = n_genes, ncol = n_cells),
    sparse = TRUE
  )
  rownames(unspliced) <- rownames(spliced)
  colnames(unspliced) <- colnames(spliced)

  # PCA for kNN moment smoothing
  pca_mat <- matrix(rnorm(n_cells * n_pcs), nrow = n_cells, ncol = n_pcs)
  rownames(pca_mat) <- colnames(spliced)

  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(spliced = spliced, unspliced = unspliced)
  )
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca_mat
  sce
}

# ---------------------------------------------------------------------------
# Test 1: velocity_moments writes assays Ms and Mu
# ---------------------------------------------------------------------------

test_that("velocity_moments writes assays Ms and Mu", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_no_gpu()

  sce    <- make_sce_for_velocity()
  sce_vm <- velocity_moments(sce,
                               n_neighbors       = 15L,
                               use_dimred        = "PCA",
                               layer_spliced     = "spliced",
                               layer_unspliced   = "unspliced")

  expect_s4_class(sce_vm, "SingleCellExperiment")

  assay_names <- SummarizedExperiment::assayNames(sce_vm)
  expect_true("Ms" %in% assay_names,
    info = paste0("velocity_moments() must produce assay 'Ms'; got: ",
                  paste(assay_names, collapse = ", ")))
  expect_true("Mu" %in% assay_names,
    info = paste0("velocity_moments() must produce assay 'Mu'; got: ",
                  paste(assay_names, collapse = ", ")))

  Ms <- SummarizedExperiment::assay(sce_vm, "Ms")
  Mu <- SummarizedExperiment::assay(sce_vm, "Mu")

  # Dimensions must match input
  expect_equal(dim(Ms), dim(SummarizedExperiment::assay(sce, "spliced")),
    info = "assay Ms must have same dimensions as spliced input")
  expect_equal(dim(Mu), dim(SummarizedExperiment::assay(sce, "unspliced")),
    info = "assay Mu must have same dimensions as unspliced input")

  # Values must be non-negative (moment-smoothed counts)
  ms_vals <- as.numeric(Ms)
  mu_vals <- as.numeric(Mu)
  expect_true(all(ms_vals[is.finite(ms_vals)] >= 0),
    info = "assay Ms must contain only non-negative finite values")
  expect_true(all(mu_vals[is.finite(mu_vals)] >= 0),
    info = "assay Mu must contain only non-negative finite values")
})

# ---------------------------------------------------------------------------
# Test 2: velocity_prep writes assays$velocity and rowData$velocity_gamma
# ---------------------------------------------------------------------------

test_that("velocity_prep writes assays$velocity and rowData$velocity_gamma", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_no_gpu()

  sce    <- make_sce_for_velocity()
  sce_vp <- velocity_prep(sce,
                            mode             = "steady_state",
                            min_S_count      = 5L,
                            min_U_count      = 2L,
                            top_n_quantile   = 5L,
                            compute_velocity = TRUE)

  expect_s4_class(sce_vp, "SingleCellExperiment")

  # assays$velocity must be present
  assay_names <- SummarizedExperiment::assayNames(sce_vp)
  expect_true("velocity" %in% assay_names,
    info = paste0("velocity_prep() must produce assay 'velocity'; got: ",
                  paste(assay_names, collapse = ", ")))

  vel_mat <- SummarizedExperiment::assay(sce_vp, "velocity")
  expect_equal(dim(vel_mat), c(nrow(sce_vp), ncol(sce_vp)),
    info = "assay 'velocity' must be genes × cells")

  # rowData$velocity_gamma must be present
  rd <- SummarizedExperiment::rowData(sce_vp)
  expect_true("velocity_gamma" %in% colnames(rd),
    info = paste0("velocity_prep() must write rowData$velocity_gamma; ",
                  "available rowData cols: ", paste(colnames(rd), collapse = ", ")))

  gamma_vals <- as.numeric(rd$velocity_gamma)
  expect_true(all(is.finite(gamma_vals[!is.na(gamma_vals)])),
    info = "rowData$velocity_gamma must contain only finite (or NA) values")

  # Non-NA gamma must be positive (degradation rate)
  pos_gamma <- gamma_vals[!is.na(gamma_vals)]
  expect_true(all(pos_gamma >= 0),
    info = "rowData$velocity_gamma must be non-negative where not NA")

  # rowData$velocity_filter should also be present (boolean filter)
  expect_true("velocity_filter" %in% colnames(rd),
    info = "velocity_prep() must write rowData$velocity_filter")
})

# ---------------------------------------------------------------------------
# Test 3: velocity_prep gamma vs velociraptor::scvelo — Spearman rho >= 0.95
# ---------------------------------------------------------------------------

test_that("velocity_prep gamma vs velociraptor::scvelo Spearman rho >= 0.95", {
  skip_if_not_installed("velociraptor")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_no_gpu()

  sce    <- make_sce_for_velocity()

  # GPU result
  sce_gpu <- velocity_prep(sce,
                             mode             = "steady_state",
                             min_S_count      = 5L,
                             min_U_count      = 2L,
                             top_n_quantile   = 5L,
                             compute_velocity = TRUE)
  gpu_gamma <- as.numeric(
    SummarizedExperiment::rowData(sce_gpu)$velocity_gamma
  )

  # CPU reference: velociraptor::scvelo (scVelo steady-state gamma estimation)
  # velociraptor expects spliced/unspliced in assays; sce already has them
  vr_out    <- velociraptor::scvelo(sce,
                                     mode             = "steady_state",
                                     assay.X          = "spliced",
                                     assay.spliced    = "spliced",
                                     assay.unspliced  = "unspliced",
                                     use.dimred       = "PCA",
                                     BPPARAM          = BiocParallel::SerialParam())
  cpu_gamma <- as.numeric(SummarizedExperiment::rowData(vr_out)$velocity_gamma)

  # Only compare genes where both methods estimated gamma (non-NA)
  shared_ok <- !is.na(gpu_gamma) & !is.na(cpu_gamma) & is.finite(gpu_gamma) &
                 is.finite(cpu_gamma)
  if (sum(shared_ok) < 20L) {
    skip(paste0("Fewer than 20 genes with valid gamma from both methods (",
                sum(shared_ok), ") — skip Spearman check"))
  }

  rho <- cor(gpu_gamma[shared_ok], cpu_gamma[shared_ok],
             method = "spearman", use = "complete.obs")
  tolerance <- 0.95

  expect_gte(rho, tolerance,
    info = paste0("Spearman rho (velocity_prep gamma vs velociraptor::scvelo) = ",
                  round(rho, 4), " (threshold = ", tolerance, ")"))
})
