# SPDX-License-Identifier: MIT
# Offline tests for the bundle feature->gene aggregation logic.

test_that(".bundle_aggregate_to_vocab sums features per gene onto the vocab axis", {
    skip_if_not_installed("Matrix")

    # Two genes, three features (gene A has 2 exons, gene B has 1), 2 cells.
    feat <- Matrix::sparseMatrix(
        i = c(1, 2, 3, 1, 3),
        j = c(1, 1, 1, 2, 2),
        x = c(5, 3, 7, 2, 9),
        dims = c(3, 2)
    )
    rownames(feat) <- c(
        "ENSG00000000001_GENEA_chr1:1-2",
        "ENSG00000000001_GENEA_chr1:5-6",
        "ENSG00000000002_GENEB_chr1:9-9"
    )
    feat <- methods::as(feat, "CsparseMatrix")

    vocab <- c("ENSG00000000001", "ENSG00000000002")
    agg <- singlet:::.bundle_aggregate_to_vocab(feat, vocab)

    expect_equal(dim(agg), c(2L, 2L))
    # Gene A cell1 = 5 + 3 = 8 ; gene B cell1 = 7
    expect_equal(as.numeric(agg[, 1]), c(8, 7))
    # Gene A cell2 = 2 ; gene B cell2 = 9
    expect_equal(as.numeric(agg[, 2]), c(2, 9))
})

test_that(".bundle_aggregate_to_vocab pads genes absent from the matrix", {
    skip_if_not_installed("Matrix")
    feat <- Matrix::sparseMatrix(i = 1, j = 1, x = 4, dims = c(1, 1))
    rownames(feat) <- "ENSG00000000001_GENEA_chr1:1-2"
    feat <- methods::as(feat, "CsparseMatrix")

    # vocab contains a gene that does not appear in the matrix
    vocab <- c("ENSG00000000001", "ENSG00000000099")
    agg <- singlet:::.bundle_aggregate_to_vocab(feat, vocab)

    expect_equal(dim(agg), c(2L, 1L))
    expect_equal(as.numeric(agg[, 1]), c(4, 0))
})

test_that(".bundle_aggregate_to_vocab returns NULL on NULL input", {
    expect_null(singlet:::.bundle_aggregate_to_vocab(NULL, c("ENSG1")))
})

test_that("read_singlet errors cleanly on a missing file", {
    expect_error(read_singlet("/nonexistent/GSE000000.singlet"),
                 "no such .singlet file")
})
