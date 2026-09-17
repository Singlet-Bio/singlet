# SPDX-License-Identifier: MIT
# Tests for the multi-modality accessors on .singlet bundles.
#
# These exercise the parts of the reader that do not need the native .1pz
# codec: the modality registry, member resolution across the flat and
# nested sidecar layouts, and the table/JSON/text readers. Matrix reads
# are covered by test-read_1pz.R and by the Python test-suite.

make_test_bundle <- function(dir = tempfile("bundle_")) {
    dir.create(dir, recursive = TRUE, showWarnings = FALSE)
    stage <- file.path(dir, "stage")
    sample_dir <- file.path(stage, "samples", "GSM0000001")
    dir.create(file.path(sample_dir, "donor"), recursive = TRUE)
    dir.create(file.path(sample_dir, "nonhost"), recursive = TRUE)

    writeLines(
        '{"gsm_ids": ["GSM0000001"], "gse_id": "GSE000001"}',
        file.path(stage, "manifest.json"))
    writeLines(
        paste0('{"genes": [{"gene_id": "ENSG01", "gene_name": "AAA"}], ',
               '"reference_build": "GRCh38-2024-A"}'),
        file.path(stage, "feature_vocab.json"))
    writeLines(
        c("barcode\tis_cell", "AAACCCA\tTrue", "AAACCCT\tFalse"),
        file.path(sample_dir, "cell_calls.tsv"))
    writeLines(
        '{"sample_id": "GSM0000001", "status": "DONE"}',
        file.path(sample_dir, "summary.json"))
    # Nested layout (newer bundles).
    writeLines(
        c("barcode\tdonor", "AAACCCA\tdonor0"),
        file.path(sample_dir, "donor", "donor_assignments.tsv"))
    writeLines(
        c("taxon\treads", "Escherichia coli\t42"),
        file.path(sample_dir, "nonhost", "nonhost_em_abundance.tsv"))
    # Flat layout (older bundles).
    writeLines(
        c("pos\tref\talt\taf", "3243\tA\tG\t0.12"),
        file.path(sample_dir, "mt_variants.tsv"))
    writeLines("STAR ran fine", file.path(sample_dir, "star_Log.final.out"))

    bundle <- file.path(dir, "GSE000001.singlet")
    old <- setwd(stage)
    on.exit(setwd(old), add = TRUE)
    utils::zip(bundle, list.files(".", recursive = TRUE), flags = "-rq")
    bundle
}

skip_if_no_zip <- function() {
    if (!nzchar(Sys.which("zip"))) {
        testthat::skip("the `zip` command is not available")
    }
}


test_that("the modality registry is well formed", {
    for (nm in names(SINGLET_MODALITIES)) {
        mod <- SINGLET_MODALITIES[[nm]]
        expect_true(length(mod$members) >= 1L, info = nm)
        expect_false(anyDuplicated(mod$members) > 0L, info = nm)
        expect_true(mod$kind %in% c("matrix", "table", "json", "text"), info = nm)
        expect_true(nzchar(mod$description), info = nm)
    }
})

test_that("the R registry matches the Python one", {
    # Guards against the two packages drifting apart.
    expect_true(all(c("exon_counts", "intron_counts", "junctions", "splice_psi",
                      "mt_heteroplasmy", "mt_variants", "donor_assignments",
                      "nonhost_species", "vdj_gene_usage", "summary")
                    %in% names(SINGLET_MODALITIES)))
})

test_that("singlet_files lists members relative to the sample", {
    skip_if_no_zip()
    bundle <- make_test_bundle()
    files <- singlet_files(bundle, "GSM0000001")
    expect_true("cell_calls.tsv" %in% files)
    expect_true("donor/donor_assignments.tsv" %in% files)
    expect_false(any(startsWith(files, "samples/")))

    all_members <- singlet_files(bundle)
    expect_true("manifest.json" %in% all_members)
})

test_that("singlet_modalities reports what is present", {
    skip_if_no_zip()
    bundle <- make_test_bundle()
    found <- singlet_modalities(bundle)
    expect_true(all(c("cell_calls", "summary", "donor_assignments",
                      "nonhost_species", "mt_variants", "star_log")
                    %in% names(found)))
    expect_false("junctions" %in% names(found))
    expect_true(all(nzchar(found)))
})

test_that("singlet_modalities defaults to the first sample", {
    skip_if_no_zip()
    bundle <- make_test_bundle()
    expect_identical(singlet_modalities(bundle),
                     singlet_modalities(bundle, "GSM0000001"))
})

test_that("singlet_has resolves both the flat and nested layouts", {
    skip_if_no_zip()
    bundle <- make_test_bundle()
    expect_true(singlet_has(bundle, "donor_assignments", "GSM0000001"))
    expect_true(singlet_has(bundle, "mt_variants", "GSM0000001"))
    expect_false(singlet_has(bundle, "junctions", "GSM0000001"))
    # Without a gsm the check scans every sample.
    expect_true(singlet_has(bundle, "summary"))
})

test_that("singlet_read returns the right R type per modality kind", {
    skip_if_no_zip()
    bundle <- make_test_bundle()

    json <- singlet_read(bundle, "GSM0000001", "summary")
    expect_type(json, "list")
    expect_identical(json$sample_id, "GSM0000001")

    tbl <- singlet_read(bundle, "GSM0000001", "donor_assignments")
    expect_s3_class(tbl, "data.frame")
    expect_identical(tbl$donor, "donor0")

    txt <- singlet_read(bundle, "GSM0000001", "star_log")
    expect_type(txt, "character")
    expect_match(txt, "STAR ran fine")
})

test_that("singlet_read accepts a literal member path", {
    skip_if_no_zip()
    bundle <- make_test_bundle()
    tbl <- singlet_read(bundle, "GSM0000001", "nonhost/nonhost_em_abundance.tsv")
    expect_identical(tbl$taxon, "Escherichia coli")
})

test_that("a missing modality errors with the available list", {
    skip_if_no_zip()
    bundle <- make_test_bundle()
    expect_error(singlet_read(bundle, "GSM0000001", "junctions"),
                 "not present")
})

test_that("a missing bundle errors early", {
    expect_error(singlet_files(file.path(tempdir(), "nope.singlet")),
                 "no such")
})
