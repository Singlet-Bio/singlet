# SPDX-License-Identifier: MIT
test_that("read_singlet_dir loads every .1pz and attaches metadata", {
    dir <- skip_if_no_fixture()

    dd <- read_singlet_dir(dir)
    expect_true(length(dd) > 0L)
    expect_true("gene_counts" %in% names(dd) || "spliced" %in% names(dd))

    kv <- attr(dd, "user_kv")
    expect_true(!is.null(kv))
    expect_true("gsm_id" %in% names(kv))
})

test_that("include= filters to only requested matrices", {
    dir <- skip_if_no_fixture()
    dd <- read_singlet_dir(dir, include = c("gene_counts", "spliced"))
    expect_true(length(dd) <= 2L)
    expect_true(all(names(dd) %in% c("gene_counts", "spliced")))
})

test_that("exclude= drops specific matrices", {
    dir <- skip_if_no_fixture()
    dd <- read_singlet_dir(dir, exclude = c("gene_counts"))
    expect_false("gene_counts" %in% names(dd))
})

test_that("gene_counts equals spliced + unspliced + ambiguous on real outputs", {
    # Empirical property that justifies dropping gene_counts.1pz from the
    # pipeline's NFS copy tree. If this fails on any new sample, the drop
    # policy is unsafe and must be reverted.
    dir <- skip_if_no_fixture()
    dd <- read_singlet_dir(
        dir,
        include = c("gene_counts", "spliced", "unspliced", "ambiguous")
    )
    needed <- c("gene_counts", "spliced", "unspliced", "ambiguous")
    if (!all(needed %in% names(dd))) {
        skip("fixture lacks the full velocity trio + gene_counts")
    }
    stopifnot(requireNamespace("Matrix", quietly = TRUE))
    reconstructed <- dd$spliced + dd$unspliced + dd$ambiguous
    diff <- dd$gene_counts - reconstructed
    expect_equal(length(diff@x), 0L)
})

test_that("read_singlet_dir errors on a non-directory", {
    expect_error(read_singlet_dir("/nonexistent_dir"), "not a directory")
})
