# SPDX-License-Identifier: MIT
test_that("read_1pz returns a dgCMatrix with the right shape and attributes", {
    dir <- skip_if_no_fixture()
    gc_path <- file.path(dir, "gene_counts.1pz")
    if (!file.exists(gc_path)) gc_path <- file.path(dir, "spliced.1pz")
    if (!file.exists(gc_path)) skip("fixture lacks gene_counts.1pz and spliced.1pz")

    mat <- read_1pz(gc_path)

    expect_s4_class(mat, "dgCMatrix")
    expect_gt(nrow(mat), 0L)
    expect_gt(ncol(mat), 0L)
    expect_true(!is.null(rownames(mat)))
    expect_true(!is.null(colnames(mat)))
    expect_length(rownames(mat), nrow(mat))
    expect_length(colnames(mat), ncol(mat))

    # Embedded metadata attached as an attribute
    kv <- attr(mat, "user_kv")
    expect_true(!is.null(kv))
    expect_true(is.character(kv))
    expect_true("gsm_id" %in% names(kv))
    expect_true(grepl("^GSM", kv[["gsm_id"]]))

    # Narrow-width hint is 1, 2, or 3
    vt <- attr(mat, "vt_code")
    expect_true(!is.null(vt))
    expect_true(vt %in% c(1L, 2L, 3L))
})

test_that("read_1pz errors cleanly on a missing file", {
    expect_error(read_1pz("/nonexistent/path/to/file.1pz"), "pz_reader")
})

test_that("read_1pz errors cleanly on a garbage file", {
    tmp <- tempfile(fileext = ".1pz")
    writeBin(as.raw(rep(0, 200)), tmp)
    on.exit(unlink(tmp))
    expect_error(read_1pz(tmp), "pz_reader")
})

test_that("rownames look like Ensembl gene IDs on a 10x/drop-seq fixture", {
    dir <- skip_if_no_fixture()
    gc_path <- file.path(dir, "gene_counts.1pz")
    if (!file.exists(gc_path)) gc_path <- file.path(dir, "spliced.1pz")
    if (!file.exists(gc_path)) skip("fixture lacks gene_counts.1pz and spliced.1pz")

    mat <- read_1pz(gc_path)
    # At least a few of the rownames are Ensembl IDs — the 2024-A GRCh38
    # annotation that the pipeline uses by default.
    expect_true(any(grepl("^ENSG", rownames(mat)[seq_len(min(100, nrow(mat)))])))
})
