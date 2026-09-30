# SPDX-License-Identifier: MIT
# Cell-call parsing and the handling of samples with no usable cells.
#
# None of this needs the native .1pz codec: a "hollow" sample is one whose
# count matrices are missing, which is exactly the case to exercise.

test_that("called barcodes honour every barcode column spelling", {
    for (col in c("barcode", "cb", "cell_barcode", "CB")) {
        cc <- data.frame(x = c("A", "B", "C"),
                         is_cell = c("True", "False", "true"),
                         stringsAsFactors = FALSE)
        names(cc)[1] <- col
        expect_identical(singlet:::.called_barcodes_from_table(cc), c("A", "C"),
                         info = col)
    }
})

test_that("without is_cell every listed barcode counts as called", {
    cc <- data.frame(cb = c("A", "B"), stringsAsFactors = FALSE)
    expect_identical(singlet:::.called_barcodes_from_table(cc), c("A", "B"))
})

test_that("is_cell parses logical, numeric and string spellings", {
    expect_identical(singlet:::.as_flag(c(TRUE, FALSE, NA)),
                     c(TRUE, FALSE, FALSE))
    expect_identical(singlet:::.as_flag(c(1, 0, NA)), c(TRUE, FALSE, FALSE))
    expect_identical(singlet:::.as_flag(c("True", "FALSE", "1", "0", "", NA)),
                     c(TRUE, FALSE, TRUE, FALSE, FALSE, FALSE))
})

test_that("an empty call set means no cells; no barcode column means unknown", {
    empty <- data.frame(barcode = character(0), is_cell = character(0),
                        stringsAsFactors = FALSE)
    expect_identical(singlet:::.called_barcodes_from_table(empty), character(0))
    expect_null(singlet:::.called_barcodes_from_table(
        data.frame(is_cell = "True", stringsAsFactors = FALSE)))
    expect_null(singlet:::.called_barcodes_from_table(NULL))
})

test_that("cell_calls.tsv is read as text so numeric-looking barcodes survive", {
    p <- tempfile(fileext = ".tsv")
    on.exit(unlink(p), add = TRUE)
    writeLines(c("CB\tis_cell", "001\t1", "002\t0"), p)
    expect_identical(
        singlet:::.called_barcodes_from_table(singlet:::.read_cell_calls(p)),
        "001")
    expect_null(singlet:::.read_cell_calls(file.path(tempdir(), "nope.tsv")))
})

test_that("a sample without count matrices is reported, not fatal", {
    root <- tempfile("extract_")
    dir.create(file.path(root, "samples", "GSM0000001"), recursive = TRUE)
    on.exit(unlink(root, recursive = TRUE), add = TRUE)
    writeLines(c("barcode\tis_cell", "AAAC\tTrue"),
               file.path(root, "samples", "GSM0000001", "cell_calls.tsv"))

    res <- singlet:::.bundle_load_gsm(root, "GSM0000001", "ENSG00000000001")
    expect_type(res, "character")
    expect_match(res, "no count matrix")
})

test_that("a package that cannot be loaded is reported with the real reason", {
    err <- tryCatch(
        singlet:::.require_namespaces("singletNoSuchPackage", "test()",
                                      "install.packages(\"x\")"),
        error = conditionMessage)
    expect_match(err, "could not be loaded", fixed = TRUE)
    expect_match(err, "singletNoSuchPackage", fixed = TRUE)
    expect_match(err, "install.packages(\"x\")", fixed = TRUE)
})


# A bundle whose samples have cell calls but no count matrices.
make_hollow_bundle <- function(gsms = c("GSM0000001", "GSM0000002")) {
    dir <- tempfile("hollow_")
    stage <- file.path(dir, "stage")
    for (g in gsms) {
        sd <- file.path(stage, "samples", g)
        dir.create(sd, recursive = TRUE)
        writeLines(c("barcode\tis_cell", "AAAC\tTrue"),
                   file.path(sd, "cell_calls.tsv"))
    }
    writeLines(
        sprintf('{"gse_id": "GSE000001", "gsm_ids": [%s]}',
                paste0('"', gsms, '"', collapse = ", ")),
        file.path(stage, "manifest.json"))
    writeLines(
        paste0('{"genes": [{"gene_id": "ENSG00000000001", "gene_name": "AAA"}], ',
               '"reference_build": "GRCh38-2024-A"}'),
        file.path(stage, "feature_vocab.json"))

    bundle <- file.path(dir, "GSE000001.singlet")
    old <- setwd(stage)
    on.exit(setwd(old), add = TRUE)
    utils::zip(bundle, list.files(".", recursive = TRUE), flags = "-rq")
    bundle
}

skip_if_cannot_build_sce <- function() {
    skip_if_not_installed("SingleCellExperiment")
    skip_if_not_installed("SummarizedExperiment")
    skip_if_not_installed("S4Vectors")
    if (!nzchar(Sys.which("zip"))) {
        skip("the `zip` command is not available")
    }
}

test_that("read_singlet names the reason when no sample has cells", {
    skip_if_cannot_build_sce()
    bundle <- make_hollow_bundle()
    err <- tryCatch(read_singlet(bundle), error = conditionMessage)
    expect_match(err, "contained no loadable cells", fixed = TRUE)
    expect_match(err, "GSM0000001 (no count matrix", fixed = TRUE)
    expect_match(err, "GSM0000002 (no count matrix", fixed = TRUE)
})

test_that("asking for a sample the bundle does not hold says so", {
    skip_if_cannot_build_sce()
    bundle <- make_hollow_bundle()
    err <- tryCatch(singlet:::.read_singlet_bundle(bundle, gsms = "GSM0000099"),
                    error = conditionMessage)
    expect_match(err, "GSM0000099 (not in this bundle", fixed = TRUE)
})
