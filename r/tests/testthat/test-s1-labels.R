# SPDX-License-Identifier: MIT
# Per-cell labels in colData (R/s1_labels.R): the default cell type level and score rule, categorical columns, the hidden
# unvalidated doublet score, the GEO metadata prefix, and load() / singlet_cells() on a labelled synthetic study.
#
# The fixture tests use the Python test helper with --labels (tests/python/_s1_synth.py: GSM9000001 carries the label columns,
# truth_GSM9000001_labels.tsv the expected colData):
#
#   python -P tests/python/_s1_synth.py /some/dir --labels ; export SINGLET_S1_LABELS_ROOT=/some/dir
#
# Without it (and without Python + the singlet package to build it) they skip. SINGLET_LIVE_TESTS=1 adds a read of GSE286650
# from data.singlet.bio.

T_PATH <- "CL:0000988>CL:0000542>CL:0000084>CL:0000624"

stored_cols <- function(paths, ps) {
    n <- length(paths)
    cols <- list(cell_type_path = paths)
    for (k in 1:6) cols[[paste0("cell_type_p", k)]] <- vapply(ps, function(p) if (length(p) >= k) p[[k]] else NaN, numeric(1))
    last <- vapply(strsplit(paths, ">", fixed = TRUE), function(x) if (length(x)) x[[length(x)]] else "", "")
    cols$cell_type_cl <- last
    cols$cell_type_label <- ifelse(nzchar(last), unname(singlet:::.s1_cl_names()[last]), "")
    cols$cell_type_score <- vapply(ps, function(p) if (length(p)) prod(p) else NaN, numeric(1))
    cols
}

test_that("the CL names table ships with the package", {
    nm <- singlet:::.s1_cl_names()
    expect_identical(unname(nm["CL:0000084"]), "T cell")
    expect_identical(unname(nm["CL:0000988"]), "hematopoietic cell")
})

test_that("cell_type is the deepest level <= 2 whose path score passes 0.5", {
    cols <- stored_cols(c(T_PATH, "CL:0000988>CL:0000542", "CL:0000066", ""),
                        list(c(0.9, 0.8, 0.7, 0.6), c(0.9, 0.4), 0.3, numeric(0)))
    ct <- singlet:::.s1_cell_type_columns(cols, 4L)
    expect_identical(ct$cell_type_ontology_term_id, c("CL:0000542", "CL:0000988", NA, NA))
    expect_identical(ct$cell_type[[2]], "hematopoietic cell")
    expect_identical(ct$cell_type_level, c(2L, 1L, 0L, 0L))
    expect_equal(ct$cell_type_score[1:2], c(0.72, 0.9), tolerance = 1e-6)
    expect_identical(ct$cell_type_deepest_ontology_term_id, c("CL:0000624", "CL:0000542", "CL:0000066", NA))
    expect_identical(ct$cell_type_deepest_level, c(4L, 2L, 1L, 0L))
    deep <- singlet:::.s1_cell_type_columns(cols, 4L, level = 6L, min_score = 0)
    expect_identical(deep$cell_type_ontology_term_id, c("CL:0000624", "CL:0000542", "CL:0000066", NA))
    one <- singlet:::.s1_cell_type_columns(cols, 4L, level = 1L)
    expect_identical(one$cell_type_ontology_term_id, c("CL:0000988", "CL:0000988", NA, NA))
})

test_that("label columns: order, categories, renames and the hidden doublet score", {
    cols <- c(list(barcode = c("AAA", "CCC"), called = c(TRUE, FALSE)),
              stored_cols(c(T_PATH, ""), list(c(0.9, 0.8, 0.7, 0.6), numeric(0))),
              list(sex_chrom = c("female", ""), sex_score = c(0.99, NaN), called_ed = c(TRUE, TRUE),
                   qc_doublet_score = c(0.1, 0.9), don_donor_id = c(0, 1), cc_pos = c(0, 1)))
    d <- singlet:::.s1_label_columns(cols, 2L, list(don_donor_id = c("donor0", "unassigned")))
    expect_false(any(c("barcode", "cc_pos", "qc_doublet_score", "cell_type_cl", "cell_type_label") %in% names(d)))
    expect_identical(names(d)[1:2], c("cell_type", "cell_type_ontology_term_id"))
    expect_identical(d$don_donor_id, c("donor0", "unassigned"))
    expect_identical(d$sex_chrom, c("female", NA))
    expect_true(is.na(d$sex_score[[2]]) && !is.nan(d$sex_score[[2]]))
    expect_true("qc_doublet_score" %in% names(singlet:::.s1_label_columns(cols, 2L, include_unvalidated = TRUE)))
    only <- singlet:::.s1_label_columns(cols, 2L, columns = c("cell_type", "cell_type_cl", "qc_doublet_score"))
    expect_setequal(names(only), c("cell_type", "cell_type_deepest_ontology_term_id", "qc_doublet_score"))
    cats <- singlet:::.s1_label_categories(list(
        "donor/donor_assignments.tsv" = list(kind = 4, params = list(c = list(
            list(n = "donor_id", s = "c", c = "don_donor_id", f = list(k = "cat", labels = list("donor0", "unassigned"))),
            list(n = "prob_doublet", s = "k", v = "0")))),
        "exon_counts.1pz" = list(kind = 1)))
    expect_identical(cats, list(don_donor_id = c("donor0", "unassigned")))
})

test_that("GEO sample metadata is prefixed and never collides with the per-cell labels", {
    df <- singlet:::.bundle_obs_frame("GSM1", list(organism = "Homo sapiens", enriched = list(meta_cell_type = "T cell")), 2L)
    expect_false(any(c("cell_type", "sex", "tissue") %in% names(df)))
    expect_identical(df$geo_cell_type, c("T cell", "T cell"))
})

labels_root <- local({
    cache <- NULL
    function() {
        if (!is.null(cache)) return(cache)
        env <- Sys.getenv("SINGLET_S1_LABELS_ROOT")
        if (nzchar(env) && file.exists(file.path(env, "truth_GSM9000001_labels.tsv"))) {
            cache <<- normalizePath(env, winslash = "/")
            return(cache)
        }
        py <- Sys.getenv("SINGLET_PYTHON", Sys.which("python"))
        top <- testthat::test_path("..", "..", "..")
        script <- normalizePath(file.path(top, "tests", "python", "_s1_synth.py"), mustWork = FALSE)
        if (!nzchar(py) || !file.exists(script)) return(NA_character_)
        out <- file.path(tempdir(), "s1_synth_labels")
        res <- suppressWarnings(tryCatch(
            system2(py, c("-P", shQuote(script), shQuote(out), "--labels"), stdout = TRUE, stderr = TRUE,
                    env = c(paste0("PYTHONPATH=", normalizePath(file.path(top, "python"), mustWork = FALSE)))),
            error = function(e) NULL))
        if (is.null(res) || !file.exists(file.path(out, "truth_GSM9000001_labels.tsv"))) return(NA_character_)
        cache <<- normalizePath(out, winslash = "/")
        cache
    }
})

skip_if_no_labels <- function() {
    testthat::skip_if_not_installed("Matrix")
    testthat::skip_if_not_installed("jsonlite")
    testthat::skip_if_not_installed("SingleCellExperiment")
    r <- labels_root()
    if (is.na(r)) testthat::skip("no labelled test study: python -P tests/python/_s1_synth.py <dir> --labels")
    r
}

with_labels_cache <- function(code) {
    d <- tempfile("singlet_cache_")
    dir.create(d)
    old <- Sys.getenv("SINGLET_CACHE_DIR", unset = NA)
    Sys.setenv(SINGLET_CACHE_DIR = d)
    singlet:::.s1_clear_memory()
    on.exit({
        if (is.na(old)) Sys.unsetenv("SINGLET_CACHE_DIR") else Sys.setenv(SINGLET_CACHE_DIR = old)
        unlink(d, recursive = TRUE)
        singlet:::.s1_clear_memory()
    })
    force(code)
}

test_that("load() puts the labels in colData, matching the Python client", {
    root <- skip_if_no_labels()
    with_labels_cache({
        mp <- file.path(root, "s1", "m", "GSE900001.json")
        expect_warning(sce <- load(mp), "skipped 1 of 3")
        cd <- as.data.frame(SummarizedExperiment::colData(sce), check.names = FALSE)
        truth <- utils::read.delim(file.path(root, "truth_GSM9000001_labels.tsv"), na.strings = "NA", stringsAsFactors = FALSE,
                                   quote = "", colClasses = c(barcode = "character"))
        one <- cd[cd$gsm_id == "GSM9000001", , drop = FALSE]
        expect_identical(sub("^GSM9000001_", "", rownames(one)), truth$barcode)
        for (nm in c("cell_type", "cell_type_ontology_term_id", "cell_type_deepest_ontology_term_id", "cell_type_path", "sex_chrom",
                     "ancestry_hancestro", "don_donor_id", "cyc_phase")) {
            expect_identical(as.character(one[[nm]]), as.character(truth[[nm]]), info = nm)
        }
        for (nm in c("cell_type_level", "cell_type_score", "sex_score", "ancestry_prob", "called_ed_fdr")) {
            expect_equal(as.numeric(one[[nm]]), as.numeric(truth[[nm]]), tolerance = 1e-5, info = nm)
        }
        expect_identical(one$called_ed, as.logical(truth$called_ed))
        expect_false("qc_doublet_score" %in% names(cd))
        expect_identical(names(cd)[1:3], c("gsm_id", "cell_type", "cell_type_ontology_term_id"))
        expect_true(all(c("organism", "geo_cell_type", "geo_sex") %in% names(cd)))
        three <- cd[cd$gsm_id == "GSM9000003", , drop = FALSE]
        expect_true(nrow(three) == 25L && all(is.na(three$cell_type)) && all(is.na(three$called_ed)))
        sce2 <- suppressWarnings(load(mp, cell_type_level = 6L, cell_type_min_score = 0, include_unvalidated = TRUE))
        expect_true("qc_doublet_score" %in% names(SummarizedExperiment::colData(sce2)))
        expect_identical(max(sce2$cell_type_level), 4L)
    })
})

test_that("singlet_cells() filters on labels and hides the doublet score unless asked", {
    root <- skip_if_no_labels()
    with_labels_cache({
        st <- singlet_study("GSE900001", base = root)
        fem <- singlet_cells(st, sex_chrom == "female", samples = "GSM9000001")
        expect_true(ncol(fem) > 0L && all((fem$matrix_column - 1L) %% 4L == 0L))
        expect_true(all(fem$cell_type_ontology_term_id == "CL:0000542"))
        hi <- singlet_cells(st, qc_doublet_score > 0.5, samples = "GSM9000001")
        expect_true(ncol(hi) > 0L && !"qc_doublet_score" %in% names(SummarizedExperiment::colData(hi)))
        sh <- singlet_cells(st, qc_doublet_score > 0.5, samples = "GSM9000001", columns = c("qc_doublet_score", "cell_type"))
        expect_setequal(names(SummarizedExperiment::colData(sh)), c("gsm_id", "matrix_column", "qc_doublet_score", "cell_type"))
        both <- singlet_cells(st, called, samples = c("GSM9000001", "GSM9000003"))
        expect_identical(ncol(both), 65L)
    })
})

test_that("live: GSE286650 from data.singlet.bio has the labels in colData", {
    skip_if_not(identical(Sys.getenv("SINGLET_LIVE_TESTS"), "1"), "set SINGLET_LIVE_TESTS=1 to read data.singlet.bio")
    skip_if_not_installed("SingleCellExperiment")
    skip_if_not_installed("curl")
    with_labels_cache({
        sce <- singlet_load("GSE286650")
        cd <- SummarizedExperiment::colData(sce)
        expect_true(all(c("cell_type", "cell_type_ontology_term_id", "sex_chrom", "ancestry_hancestro", "called", "called_ed",
                          "geo_cell_type") %in% names(cd)))
        expect_false("qc_doublet_score" %in% names(cd))
        expect_gt(mean(!is.na(cd$cell_type)), 0.5)
        expect_true(all(grepl("^CL:[0-9]{7}$", stats::na.omit(cd$cell_type_ontology_term_id))))
        expect_gt(mean(cd$sex_chrom == "male", na.rm = TRUE), 0.5)
    })
})
