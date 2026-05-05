test_that("as_sce builds a SingleCellExperiment with the right assays", {
    dir <- skip_if_no_fixture()
    skip_if_not_installed("SingleCellExperiment")

    sce <- as_sce(dir)
    expect_s4_class(sce, "SingleCellExperiment")
    expect_true("counts" %in% SummarizedExperiment::assayNames(sce))

    # Metadata should carry the embedded GEO context
    md <- SummarizedExperiment::metadata(sce)
    expect_true(!is.null(md$singlify))
    expect_true("gsm_id" %in% names(md$singlify))
})

test_that("as_sce attaches per-cell sidecars into colData", {
    dir <- skip_if_no_fixture()
    skip_if_not_installed("SingleCellExperiment")
    sce <- as_sce(dir)

    # cell_qc_metrics.tsv populates these columns when present
    cd <- SummarizedExperiment::colData(sce)
    sidecar_cols <- grep("^cell_qc_metrics__", colnames(cd), value = TRUE)
    if (length(sidecar_cols) == 0L) {
        skip("fixture lacks cell_qc_metrics.tsv")
    }
    expect_true("cell_qc_metrics__total_umis" %in% colnames(cd))
})

test_that("as_sce puts per-feature matrices into altExps", {
    dir <- skip_if_no_fixture()
    skip_if_not_installed("SingleCellExperiment")
    sce <- as_sce(dir)

    alts <- SingleCellExperiment::altExpNames(sce)
    # At least one of the per-feature matrices should be an altExp.
    # The exact set depends on which outputs the pipeline wrote.
    possible <- c("exon_counts", "intron_counts", "sj_counts",
                  "splice_psi", "mt_heteroplasmy", "vdj_gene_usage")
    expect_true(length(intersect(alts, possible)) > 0L)
})

test_that("as_seurat builds a Seurat object with velocity assays", {
    dir <- skip_if_no_fixture()
    skip_if_not_installed("Seurat")

    obj <- as_seurat(dir)
    expect_s4_class(obj, "Seurat")

    # RNA is the default assay
    expect_true("RNA" %in% Seurat::Assays(obj))

    # spliced / unspliced assays are present (scvelo-compatible)
    # at least one of them must be present; the specific one depends on
    # which matrix was chosen as the primary
    velo_candidates <- intersect(Seurat::Assays(obj),
                                 c("spliced", "unspliced", "ambiguous"))
    # Even if spliced was used as primary, unspliced should still be an
    # extra assay. (Drop-seq has ambiguous nnz=0 which we still attach.)
    expect_true(length(velo_candidates) >= 1L)

    # Embedded metadata is in @misc
    expect_true(!is.null(obj@misc$singlify))
    expect_true("gsm_id" %in% names(obj@misc$singlify))
})
