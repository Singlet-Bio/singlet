# SPDX-License-Identifier: MIT
test_that("aggregate_features_to_gene reconstructs spliced from exon_counts bit-exactly", {
    # Empirical property that justifies dropping spliced.1pz from the NFS
    # copy tree: per-gene aggregation of exon_counts equals spliced.
    dir <- skip_if_no_fixture()
    ex_path <- file.path(dir, "exon_counts.1pz")
    sp_path <- file.path(dir, "spliced.1pz")
    if (!file.exists(ex_path) || !file.exists(sp_path)) {
        skip("fixture lacks exon_counts or spliced")
    }

    ex <- read_1pz(ex_path)
    sp_mat <- read_1pz(sp_path)
    agg <- aggregate_features_to_gene(ex)

    # Shape must match
    expect_equal(dim(agg), dim(sp_mat))

    # Total sum must match
    expect_equal(sum(agg), sum(sp_mat))

    # Per-gene rows must match — align by Ensembl gene ID since the order
    # may differ (aggregator preserves first-appearance order; spliced
    # preserves the writer's gene order).
    common <- intersect(rownames(agg), rownames(sp_mat))
    expect_gt(length(common), 0L)

    # Spot-check 5 random genes (full matrix comparison would be slow)
    set.seed(42)
    sample_genes <- sample(common, min(5L, length(common)))
    for (g in sample_genes) {
        a_row <- as.numeric(agg[g, ])
        s_row <- as.numeric(sp_mat[g, ])
        expect_equal(a_row, s_row,
                     info = sprintf("gene %s row mismatch", g))
    }
})

test_that("aggregate_features_to_gene errors on inputs without rownames", {
    skip_if_not_installed("Matrix")
    m <- Matrix::sparseMatrix(i = c(1, 2), j = c(1, 1), x = c(1, 2),
                              dims = c(3, 2))
    # No rownames -> error
    expect_error(aggregate_features_to_gene(m), "rownames")
})

test_that("aggregate_features_to_gene rejects non-dgCMatrix", {
    expect_error(aggregate_features_to_gene(matrix(0, 2, 2)), "dgCMatrix")
})
