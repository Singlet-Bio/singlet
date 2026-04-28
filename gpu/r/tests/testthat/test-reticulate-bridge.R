# test-reticulate-bridge.R
# Tests for the reticulate bridge functions (rapids_harmony, rapids_umap, etc.)
# Design ref: singlet-gpu/state/designs/24-r-wrapper-foundation.md §API surface
#
# These functions are OPT-IN — they require:
#   1. reticulate R package
#   2. singlet_gpu Python wheel installed in the active Python environment
#
# Tests:
#   1. rapids_harmony errors gracefully when reticulate is unavailable
#   2. rapids_harmony works end-to-end if reticulate + singlet_gpu Python wheel present

# ---------------------------------------------------------------------------
# Helper: check if singlet_gpu Python wheel is importable via reticulate
# ---------------------------------------------------------------------------

.singlet_gpu_python_available <- function() {
  if (!requireNamespace("reticulate", quietly = TRUE)) return(FALSE)
  tryCatch(
    { reticulate::import("singlet_gpu"); TRUE },
    error = function(e) FALSE
  )
}

# ---------------------------------------------------------------------------
# Test 1: graceful failure when reticulate is absent
# ---------------------------------------------------------------------------

test_that("rapids_harmony errors gracefully if reticulate unavailable", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SingleCellExperiment")

  # Mock absence: if reticulate IS installed, we force the error-path by
  # passing a deliberately invalid SCE that causes bridge pre-flight to fail,
  # OR by unloading reticulate temporarily with a mock approach.
  # The safest portable test: call rapids_harmony with an empty SCE and no
  # reducedDim("PCA") — the bridge must abort before hitting Python.

  sce_empty <- SingleCellExperiment::SingleCellExperiment()

  if (!requireNamespace("reticulate", quietly = TRUE)) {
    # reticulate is genuinely absent — expect informative stop()
    expect_error(
      rapids_harmony(sce_empty, batch_key = "donor"),
      regexp = "reticulate",
      info = "rapids_harmony must mention 'reticulate' in its error when package absent"
    )
  } else if (!.singlet_gpu_python_available()) {
    # reticulate present but Python wheel missing
    expect_error(
      rapids_harmony(sce_empty, batch_key = "donor"),
      regexp = "singlet.gpu|singlet_gpu|reticulate",
      info = "rapids_harmony must error with install instructions when Python wheel absent"
    )
  } else {
    # Both reticulate + wheel present — skip this graceful-failure test;
    # it is meaningless when the environment is fully configured.
    testthat::skip("reticulate + singlet_gpu Python wheel available; skipping failure-mode test")
  }
})

# ---------------------------------------------------------------------------
# Test 2: end-to-end smoke when reticulate + singlet_gpu wheel are present
# ---------------------------------------------------------------------------

test_that("rapids_harmony works if reticulate + singlet_gpu Python wheel installed", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("SummarizedExperiment")
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("reticulate")
  skip_if_no_gpu()

  if (!.singlet_gpu_python_available()) {
    testthat::skip("singlet_gpu Python wheel not importable via reticulate; skipping bridge smoke test")
  }

  # Build a tiny SCE with a batch covariate and a PCA embedding
  sce <- make_tiny_sce(seed = 0xC0FFEE)
  sce_norm <- lognorm(sce)
  sce_pca  <- run_pca(sce_norm, n_pcs = 10L, seed = 0xC0FFEE)

  # Add a synthetic batch covariate (two balanced batches)
  n_cells <- ncol(sce_pca)
  SummarizedExperiment::colData(sce_pca)$donor <-
    rep(c("donor_A", "donor_B"), length.out = n_cells)

  # Run rapids_harmony batch integration
  sce_harmony <- rapids_harmony(sce_pca, batch_key = "donor")

  expect_s4_class(sce_harmony, "SingleCellExperiment")

  # rapids_harmony must add a "Harmony" (or "X_pca_harmony") reducedDim
  rdim_names <- SingleCellExperiment::reducedDimNames(sce_harmony)
  harmony_present <- any(grepl("harmony|Harmony|HARMONY", rdim_names))
  expect_true(harmony_present,
    info = paste0("Expected a Harmony embedding in reducedDims; found: ",
                  paste(rdim_names, collapse = ", ")))

  # Harmony embedding dimensions: same cells, same n_pcs
  harmony_name <- rdim_names[grepl("harmony|Harmony|HARMONY", rdim_names)][1L]
  h_embed <- SingleCellExperiment::reducedDim(sce_harmony, harmony_name)
  expect_equal(nrow(h_embed), n_cells,
    info = "Harmony embedding rows must equal number of cells")
  expect_true(ncol(h_embed) <= 10L,
    info = "Harmony embedding should not exceed input PCA dimensionality")
  expect_true(all(is.finite(h_embed)),
    info = "Harmony embedding must contain only finite values")
})
