# test-load.R
# Tests for read_1pz_sce() — the .1pz → SingleCellExperiment loader.
# Design ref: singlet-gpu/state/designs/24-r-wrapper-foundation.md §API surface
#
# Tests:
#   1. Return type is SingleCellExperiment
#   2. GEO metadata is embedded in metadata(sce)$singlify
#   3. Dimensions match the expected gene × cell counts
#   4. Real-data smoke test on GSM4037629

test_that("read_1pz_sce returns SingleCellExperiment", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("Matrix")
  skip_if_no_test_data()

  sce <- read_1pz_sce(tiny_test_path())

  expect_s4_class(sce, "SingleCellExperiment")

  # assays(sce)$counts must be a dgCMatrix (CSC) — not dense
  expect_true("counts" %in% SummarizedExperiment::assayNames(sce))
  counts_mat <- SummarizedExperiment::assay(sce, "counts")
  expect_s4_class(counts_mat, "dgCMatrix")
})

test_that("read_1pz_sce embeds GEO metadata in metadata(sce)$singlify", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_test_data()

  sce <- read_1pz_sce(tiny_test_path())

  meta <- S4Vectors::metadata(sce)
  expect_true("singlify" %in% names(meta),
    info = "metadata(sce)$singlify must be present")

  singlify_meta <- meta$singlify
  expect_type(singlify_meta, "list")

  # Required GEO fields (see CLAUDE.md §GEO Metadata Embedding)
  required_fields <- c(
    "gsm_id", "gse_id", "organism", "taxon_id",
    "protocol", "modality", "singlify_version"
  )
  for (field in required_fields) {
    expect_true(field %in% names(singlify_meta),
      info = paste0("metadata(sce)$singlify must contain field '", field, "'"))
  }
})

test_that("read_1pz_sce dimensions match expected (tiny fixture: 500 genes x 200 cells)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_test_data()

  sce <- read_1pz_sce(tiny_test_path())
  dims <- dim(sce)

  expected <- tiny_expected_dims()   # c(500L, 200L)

  expect_equal(dims[1L], expected[1L],
    info = paste0("Expected ", expected[1L], " genes, got ", dims[1L]))
  expect_equal(dims[2L], expected[2L],
    info = paste0("Expected ", expected[2L], " cells, got ", dims[2L]))

  # Row/colnames must be present (gene symbols + barcodes from .1pz TLV)
  expect_false(is.null(rownames(sce)),
    info = "rownames (gene symbols) must be embedded from .1pz TLV")
  expect_false(is.null(colnames(sce)),
    info = "colnames (cell barcodes) must be embedded from .1pz TLV")
  expect_length(rownames(sce), expected[1L])
  expect_length(colnames(sce), expected[2L])
})

test_that("read_1pz_sce GSM4037629 real-data smoke (11,560 cells, full artifact suite)", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_no_gsm4037629()
  skip_if_no_gpu()

  gsm_dir <- gsm4037629_path()

  # Default modality = "exon" maps to exon_counts.1pz
  sce <- read_1pz_sce(gsm_dir, modality = "exon")

  expect_s4_class(sce, "SingleCellExperiment")

  # Cell count: 11,560 as called by singlify EmptyDrops
  n_cells <- ncol(sce)
  expect_gt(n_cells, 10000L,
    info = paste0("GSM4037629 should have >10k cells; got ", n_cells))
  expect_equal(n_cells, gsm4037629_expected_cells(),
    info = paste0("Expected exactly 11,560 cells, got ", n_cells))

  # GEO metadata must identify this sample
  meta_sl <- S4Vectors::metadata(sce)$singlify
  expect_equal(meta_sl$gsm_id, "GSM4037629")
  expect_equal(meta_sl$gse_id, "GSE127918")
  expect_equal(meta_sl$organism, "Homo sapiens")

  # counts must be non-trivially sparse (at least 50% zero)
  counts_mat <- SummarizedExperiment::assay(sce, "counts")
  sparsity <- 1 - Matrix::nnzero(counts_mat) / prod(dim(counts_mat))
  expect_gt(sparsity, 0.5,
    info = paste0("Expected sparse counts matrix, sparsity = ", round(sparsity, 3)))

  # intron_counts.1pz modality switch
  sce_intron <- read_1pz_sce(gsm_dir, modality = "intron")
  expect_equal(dim(sce_intron), dim(sce),
    info = "intron modality should have same dims as exon")
})
