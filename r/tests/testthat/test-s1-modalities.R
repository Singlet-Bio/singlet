# SPDX-License-Identifier: MIT
# load(modalities = ) / singlet_cells(modalities = ): only the members a modality needs are read, and the result holds just
# those (counts / spliced / unspliced assays, splice_junctions / psi / vdj altExps).
#
# Uses the synthetic study of tests/python/_s1_synth.py built with --extras (junction and V(D)J members):
#
#   python -P tests/python/_s1_synth.py /some/dir --extras ; export SINGLET_S1_EXTRAS_ROOT=/some/dir
#
# If SINGLET_S1_EXTRAS_ROOT is unset the helper builds it with `python` (SINGLET_PYTHON for another interpreter).

storex_root <- local({
    cache <- NULL
    function() {
        if (!is.null(cache)) return(cache)
        env <- Sys.getenv("SINGLET_S1_EXTRAS_ROOT")
        if (nzchar(env) && dir.exists(file.path(env, "store"))) {
            cache <<- normalizePath(env, winslash = "/")
            return(cache)
        }
        py <- Sys.getenv("SINGLET_PYTHON", Sys.which("python"))
        top <- testthat::test_path("..", "..", "..")
        script <- normalizePath(file.path(top, "tests", "python", "_s1_synth.py"), mustWork = FALSE)
        if (!nzchar(py) || !file.exists(script)) return(NA_character_)
        out <- file.path(tempdir(), "store_synth_extras")
        res <- suppressWarnings(tryCatch(
            system2(py, c("-P", shQuote(script), shQuote(out), "--extras"), stdout = TRUE, stderr = TRUE,
                    env = c(paste0("PYTHONPATH=", normalizePath(file.path(top, "python"), mustWork = FALSE)))),
            error = function(e) NULL))
        if (is.null(res) || !file.exists(file.path(out, "truth_GSM9000001_sj.mtx"))) return(NA_character_)
        cache <<- normalizePath(out, winslash = "/")
        cache
    }
})

skip_if_no_s1x <- function() {
    testthat::skip_if_not_installed("Matrix")
    testthat::skip_if_not_installed("jsonlite")
    testthat::skip_if_not_installed("SingleCellExperiment")
    r <- storex_root()
    if (is.na(r)) testthat::skip("no store test study with extras: python -P tests/python/_s1_synth.py <dir> --extras")
    r
}

X_GSE <- "GSE900001"
X_G1 <- "GSM9000001"
X_G3 <- "GSM9000003"

with_x_cache <- function(code) {
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

# Bytes of sample objects read for one sample (the manifest and packs are read before the reset).
sample_bytes <- function(st, gsm, modalities) {
    singlet:::.s1_stats_reset()
    sce <- singlet:::.s1_read_study(st, gsm, modalities = modalities)
    list(sce = sce, bytes = singlet:::.s1_stats()$bytes)
}

test_that("the vocabulary normalises aliases and refuses unknown or absent modalities", {
    expect_identical(singlet:::.s1_normalize_modalities(NULL), c("counts", "spliced", "unspliced"))
    expect_identical(singlet:::.s1_normalize_modalities(NULL, layers = FALSE), "counts")
    expect_identical(singlet:::.s1_normalize_modalities(c("sj", "SPLICED", "junctions")), c("spliced", "splice_junctions"))
    expect_error(singlet:::.s1_normalize_modalities("atac"), "unknown modality.*vocabulary is counts")
    root <- skip_if_no_s1x()
    with_x_cache({
        st <- singlet_study(X_GSE, base = root)
        expect_identical(singlet_study_modalities(st), c("counts", "spliced", "unspliced", "splice_junctions", "psi", "vdj"))
        expect_identical(singlet_study_modalities(st, X_G3), c("counts", "spliced", "splice_junctions", "psi"))
        expect_error(singlet:::.s1_read_study(st, X_G1, modalities = "mt"), "no requested sample has mt.*splice_junctions")
        expect_error(singlet:::.s1_read_study(st, X_G3, modalities = "unspliced"), "GSM9000003 has counts, spliced")
    })
})

test_that("each modality reads only the cells table and its own members", {
    root <- skip_if_no_s1x()
    with_x_cache({
        st <- singlet_study(X_GSE, base = root)
        invisible(sample_bytes(st, X_G1, NULL))  # packs first, so the byte counts below are sample objects only
        mem <- st$samples[[X_G1]]$members
        len <- function(...) sum(vapply(c("cells", ...), function(n) as.numeric(mem[[n]]$len), numeric(1)))
        wants <- list(counts = c("exon_counts.1pz", "intron_counts.1pz"), spliced = "exon_counts.1pz",
                      unspliced = "intron_counts.1pz", splice_junctions = "sj_counts.1pz", vdj = "vdj_gene_usage.1pz")
        for (k in names(wants)) {
            got <- sample_bytes(st, X_G1, k)$bytes
            exact <- do.call(len, as.list(wants[[k]]))
            expect_gte(got, exact)
            expect_lte(got, exact + 64 * length(wants[[k]]))  # alignment padding between adjacent members
        }
        expect_equal(sample_bytes(st, X_G1, NULL)$bytes, sample_bytes(st, X_G1, "counts")$bytes)
    })
})

test_that("assays and altExps hold the selected modalities with the right values", {
    root <- skip_if_no_s1x()
    with_x_cache({
        st <- singlet_study(X_GSE, base = root)
        sp <- singlet:::.s1_read_study(st, X_G1, modalities = "spliced")
        expect_identical(SummarizedExperiment::assayNames(sp), "spliced")
        tr <- Matrix::readMM(file.path(root, sprintf("truth_%s_spliced.mtx", X_G1)))
        expect_equal(sum(abs(SummarizedExperiment::assay(sp, "spliced") - tr)), 0)
        expect_warning(sce <- singlet:::.s1_read_study(st, c(X_G1, X_G3), modalities = c("counts", "splice_junctions", "psi", "vdj")),
                       "vdj: GSM9000003 have no vdj_gene_usage.1pz")
        expect_identical(SummarizedExperiment::assayNames(sce), "counts")
        expect_identical(SingleCellExperiment::altExpNames(sce), c("splice_junctions", "psi", "vdj"))
        sj <- SummarizedExperiment::assay(SingleCellExperiment::altExp(sce, "splice_junctions"))
        expect_equal(ncol(sj), 65L)
        names1 <- readLines(file.path(root, sprintf("truth_%s_sj_names.txt", X_G1)))
        names3 <- readLines(file.path(root, sprintf("truth_%s_sj_names.txt", X_G3)))
        expect_identical(rownames(sj), unique(c(names1, names3)))
        i1 <- sce$gsm_id == X_G1
        t1 <- Matrix::readMM(file.path(root, sprintf("truth_%s_sj.mtx", X_G1)))
        expect_equal(sum(abs(sj[names1, i1] - t1)), 0)
        psi <- SummarizedExperiment::assay(SingleCellExperiment::altExp(sce, "psi"))
        expect_true(all(psi@x > 0 & psi@x <= 1))
        vdj <- SummarizedExperiment::assay(SingleCellExperiment::altExp(sce, "vdj"))
        expect_equal(sum(vdj[, !i1]), 0)
        expect_gt(sum(vdj[, i1]), 0)
        # without a gene modality the gene axis is empty and the data are in the altExp
        only <- singlet:::.s1_read_study(st, X_G1, modalities = "splice_junctions")
        expect_equal(nrow(only), 0L)
        expect_equal(ncol(only), 40L)
    })
})

test_that("PSI in R matches the Python reference recipe", {
    root <- skip_if_no_s1x()
    py <- Sys.getenv("SINGLET_PYTHON", Sys.which("python"))
    skip_if(!nzchar(py), "no python")
    with_x_cache({
        st <- singlet_study(X_GSE, base = root)
        sce <- singlet:::.s1_read_study(st, X_G1, modalities = "psi", cell_order = "column")
        psi <- SummarizedExperiment::assay(SingleCellExperiment::altExp(sce, "psi"))
        f <- tempfile(fileext = ".mtx")
        top <- normalizePath(testthat::test_path("..", "..", ".."), winslash = "/")
        code <- sprintf(paste(
            "import sys, numpy as np, scipy.io, scipy.sparse as sp",
            "sys.path[:0] = ['%s/python', '%s/tests/python']",
            "from singlet.s1.recipes_ctx import Matrix",
            "from singlet.s1.recipes_psi import compute_psi",
            "m = scipy.io.mmread('%s/truth_GSM9000001_sj.mtx').tocsc(); m.sort_indices()",
            "names = open('%s/truth_GSM9000001_sj_names.txt').read().split()",
            "ip, rr, v, _ = compute_psi(Matrix(m.indptr.astype(np.int64), m.indices.astype(np.int64), m.data, m.shape, names, None, None))",
            "scipy.io.mmwrite('%s', sp.csc_matrix((v.astype(np.float64), rr, ip), shape=m.shape))", sep = "\n"),
            top, top, root, root, gsub("\\\\", "/", f))
        script <- tempfile(fileext = ".py")
        writeLines(code, script)
        res <- suppressWarnings(system2(py, c("-P", shQuote(script)), stdout = TRUE, stderr = TRUE))
        skip_if(!file.exists(f), paste(res, collapse = "\n"))
        want <- methods::as(Matrix::readMM(f), "CsparseMatrix")
        # truth columns are in sorted-barcode order; the R result is in column order: compare as sets of columns
        bcs <- readLines(file.path(root, sprintf("truth_%s_barcodes.txt", X_G1)))
        got <- psi[, match(paste0(X_G1, "_", bcs), colnames(psi))]
        expect_equal(max(abs(as.matrix(got) - as.matrix(want))), 0, tolerance = 1e-6)
    })
})

test_that("singlet_cells(modalities = ) slices one member", {
    root <- skip_if_no_s1x()
    with_x_cache({
        st <- singlet_study(X_GSE, base = root)
        sub <- singlet_cells(st, n_umi_exon > 0 & called, samples = X_G1, modalities = "splice_junctions")
        expect_equal(nrow(sub), 0L)
        expect_equal(ncol(sub), 40L)
        expect_identical(SingleCellExperiment::altExpNames(sub), "splice_junctions")
        expect_identical(S4Vectors::metadata(sub)$singlet_s1$modalities, "splice_junctions")
    })
})
