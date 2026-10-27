# SPDX-License-Identifier: MIT
# The .s1 read path: manifest, ranged member reads, packs, cells tables, the C++ decoder, load(), singlet_cells(), download().
#
# The tests need a per-sample study. They use one built by the Python test helper (tests/python/_s1_synth.py: real codec, real
# vocabulary packs, three samples, plus the expected gene-level matrices as MatrixMarket files):
#
#   python -P tests/python/_s1_synth.py /some/dir ; export SINGLET_S1_TEST_ROOT=/some/dir
#
# If SINGLET_S1_TEST_ROOT is unset the helper tries to build it with `python` (set SINGLET_PYTHON for another interpreter);
# without Python or the singlet package the tests skip. SINGLET_S1_TEST_URL (a server of that directory with Range support,
# e.g. tests/python/_s1_range_server.py) enables the HTTP tests.

s1_root <- local({
    cache <- NULL
    function() {
        if (!is.null(cache)) return(cache)
        env <- Sys.getenv("SINGLET_S1_TEST_ROOT")
        if (nzchar(env) && dir.exists(file.path(env, "s1"))) {
            cache <<- normalizePath(env, winslash = "/")
            return(cache)
        }
        py <- Sys.getenv("SINGLET_PYTHON", Sys.which("python"))
        script <- normalizePath(file.path(testthat::test_path("..", "..", ".."), "tests", "python", "_s1_synth.py"), mustWork = FALSE)
        if (!nzchar(py) || !file.exists(script)) return(NA_character_)
        out <- file.path(tempdir(), "store_synth")
        res <- suppressWarnings(tryCatch(
            system2(py, c("-P", shQuote(script), shQuote(out)), stdout = TRUE, stderr = TRUE,
                    env = c(paste0("PYTHONPATH=", normalizePath(file.path(testthat::test_path("..", "..", ".."), "python"), mustWork = FALSE)))),
            error = function(e) NULL))
        if (is.null(res) || !dir.exists(file.path(out, "s1"))) return(NA_character_)
        cache <<- normalizePath(out, winslash = "/")
        cache
    }
})

skip_if_no_s1 <- function() {
    testthat::skip_if_not_installed("Matrix")
    testthat::skip_if_not_installed("jsonlite")
    r <- s1_root()
    if (is.na(r)) {
        testthat::skip("no store test study: python -P tests/python/_s1_synth.py <dir> and set SINGLET_S1_TEST_ROOT")
    }
    r
}

read_truth <- function(root, gsm, what = "") {
    m <- Matrix::readMM(file.path(root, sprintf("truth_%s%s.mtx", gsm, what)))
    methods::as(m, "CsparseMatrix")
}

GSE <- "GSE900001"
G1 <- "GSM9000001"
G2 <- "GSM9000002"
G3 <- "GSM9000003"

with_s1_cache <- function(code) {
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


test_that("the codec's instruction-set report and sha256 are available", {
    expect_match(singlet:::pz2_isa_report_(), "selected=")
    expect_identical(singlet:::pz2_sha256_(charToRaw("abc")),
                     "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
})

test_that("the manifest opens from a local directory and describes the samples", {
    root <- skip_if_no_s1()
    with_s1_cache({
        st <- singlet_study(GSE, base = root)
        expect_s3_class(st, "singlet_study")
        expect_identical(st$gsm_ids, c(G1, G2, G3))
        expect_identical(st$reference_build, "GRCh38-2024-A")
        expect_match(st$samples[[G1]]$key, "^s1/s/GSM9000001/[0-9a-f]{32}\\.s1$")
        expect_output(print(st), "singlet_study GSE900001")
        expect_error(singlet_study("GSE1", base = root), "has not been migrated")
        expect_error(singlet_study("GSE1", base = root), "legacy = TRUE")
        expect_error(singlet_study("nonsense", base = root), "not a GEO Series")
    })
})

test_that("packs give the gene table and the row -> gene map without string work", {
    root <- skip_if_no_s1()
    with_s1_cache({
        st <- singlet_study(GSE, base = root)
        genes <- strsplit(readLines(file.path(root, "genes.txt")), "\t")
        ex <- singlet:::.s1_vocab(st, st$packs$exon)
        expect_identical(ex$gene_id, vapply(genes, `[`, "", 1L))
        expect_identical(ex$gene_name, vapply(genes, `[`, "", 2L))
        expect_length(ex$row_gene, 310797L)
        expect_identical(range(ex$row_gene), c(0L, length(ex$gene_id) - 1L))
        expect_true(all(diff(ex$row_gene) >= 0L))  # exon rows are grouped by gene
        it <- singlet:::.s1_vocab(st, st$packs$intron)
        expect_length(it$row_gene, 272191L)
        expect_identical(it$gene_id, ex$gene_id)
        # the pack is cached on disk and verified by its sha256 when read again
        again <- singlet:::.s1_pack(st, st$packs$exon)
        expect_identical(startsWith(singlet:::pz2_sha256_(again), st$packs$exon), TRUE)
    })
})

test_that("load() of a local manifest reproduces the expected gene-level matrices", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    with_s1_cache({
        mp <- file.path(root, "s1", "m", paste0(GSE, ".json"))
        expect_warning(sce <- load(mp), "skipped 1 of 3")
        expect_s4_class(sce, "SingleCellExperiment")
        expect_identical(sort(unique(sce$gsm_id)), c(G1, G3))
        bcs1 <- readLines(file.path(root, sprintf("truth_%s_barcodes.txt", G1)))
        i1 <- sce$gsm_id == G1
        expect_identical(colnames(sce)[i1], paste0(G1, "_", bcs1))
        x1 <- SummarizedExperiment::assay(sce, "counts")[, i1]
        truth <- read_truth(root, G1)
        expect_equal(dim(x1), dim(truth))
        expect_equal(sum(abs(x1 - truth)), 0)
        sp1 <- SummarizedExperiment::assay(sce, "spliced")[, i1]
        expect_equal(sum(abs(sp1 - read_truth(root, G1, "_spliced"))), 0)
        un1 <- SummarizedExperiment::assay(sce, "unspliced")[, i1]
        expect_equal(sum(abs(un1 - read_truth(root, G1, "_unspliced"))), 0)
        # the sample without an intron matrix gets an all-zero unspliced assay (as the legacy reader does)
        i3 <- sce$gsm_id == G3
        expect_equal(sum(SummarizedExperiment::assay(sce, "unspliced")[, i3]), 0)
        expect_equal(sum(abs(SummarizedExperiment::assay(sce, "counts")[, i3] - read_truth(root, G3))), 0)
        expect_identical(rownames(sce)[1:2], strsplit(readLines(file.path(root, "genes.txt"))[1:2], "\t") |> vapply(`[`, "", 1L))
        sk <- S4Vectors::metadata(sce)$skipped_samples
        expect_identical(sk$gsm_id, G2)
        expect_identical(sk$reason, "0 cells called")
        expect_identical(S4Vectors::metadata(sce)$study_meta$gse_id, GSE)
        expect_identical(sce$organism[[1L]], "Homo sapiens")
    })
})

test_that("a GSM of the study loads alone; a study without usable cells says why", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    with_s1_cache({
        mp <- file.path(root, "s1", "m", paste0(GSE, ".json"))
        st <- singlet_study(GSE, base = root)
        sce <- singlet:::.s1_read_study(st, G3)
        expect_identical(unique(sce$gsm_id), G3)
        expect_error(suppressWarnings(singlet:::.s1_read_study(st, G2)), "contained no loadable cells")
    })
})

test_that("the codec decodes the same members in feature space and checks digests", {
    root <- skip_if_no_s1()
    with_s1_cache({
        st <- singlet_study(GSE, base = root)
        sm <- st$samples[[G1]]
        m <- sm$members[["exon_counts.1pz"]]
        raw <- singlet:::.s1_reader(st, sm)(m$off, m$len)[[1L]]
        hdr <- singlet:::pz2_header_(raw)
        expect_identical(hdr$digest, m$digest)
        expect_equal(c(hdr$m, hdr$n), c(310797, 300))
        res <- singlet:::pz2_decode_(raw, numeric(0), list(), 0, hdr$n, "canonical", NULL, NULL, 0L, 2L, "auto")
        x <- singlet:::.s1_dgc(res)
        expect_equal(dim(x), c(310797L, 300L))
        expect_equal(length(x@x), hdr$nnz)
        expect_true(all(x@x > 0))
        expect_error(singlet:::pz2_header_(raw[1:100]), "256")
        # all ISA paths decode the same bytes
        a <- singlet:::pz2_decode_(raw, numeric(0), list(), 0, 100, "canonical", NULL, NULL, 0L, 1L, "scalar")
        b <- singlet:::pz2_decode_(raw, numeric(0), list(), 0, 100, "canonical", NULL, NULL, 0L, 1L, "auto")
        expect_identical(a$i, b$i)
        expect_identical(a$x, b$x)
    })
})

test_that("singlet_cells() returns just the cells that satisfy the condition", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    with_s1_cache({
        all1 <- suppressWarnings(singlet:::.s1_read_study(singlet_study(GSE, base = root), G1))
        sub <- singlet_cells(singlet_study(GSE, base = root), n_umi_exon >= 0 & called, samples = G1)
        expect_equal(ncol(sub), ncol(all1))
        expect_true(all(sub$called))
        expect_true(all(c("gsm_id", "matrix_column", "n_umi_exon") %in% colnames(SummarizedExperiment::colData(sub))))
        none <- singlet_cells(singlet_study(GSE, base = root), n_umi_exon > 1e12, samples = G1)
        expect_equal(ncol(none), 0L)
        few <- singlet_cells(singlet_study(GSE, base = root), NULL, samples = G1, max_cells = 5)
        expect_equal(ncol(few), 5L)
        expect_identical(SummarizedExperiment::colData(few)$matrix_column, 1:5)
    })
})

test_that("download() mirrors a study and load() then reads the local copy", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    with_s1_cache({
        old <- Sys.getenv("SINGLET_DATA_BASE", unset = NA)
        Sys.setenv(SINGLET_DATA_BASE = root)
        on.exit(if (is.na(old)) Sys.unsetenv("SINGLET_DATA_BASE") else Sys.setenv(SINGLET_DATA_BASE = old), add = TRUE)
        mp <- download(GSE)
        expect_true(file.exists(mp))
        expect_true(all(file.exists(file.path(dirname(dirname(mp)), "s", c(G1, G2, G3)))))
        expect_true(length(list.files(file.path(dirname(dirname(mp)), "ref"), pattern = "\\.pack$")) >= 2L)
        expect_error(download("GSE1"), "legacy = TRUE")
        sce <- suppressWarnings(load(GSE))
        expect_equal(ncol(sce), 40L + 25L)
    })
})

test_that("ranged HTTP reads work against a server that honours Range", {
    skip_if_no_s1()
    url <- Sys.getenv("SINGLET_S1_TEST_URL")
    skip_if(!nzchar(url), "set SINGLET_S1_TEST_URL to a Range-capable server of the test study")
    skip_if_not_installed("curl")
    skip_if_not_installed("SingleCellExperiment")
    with_s1_cache({
        singlet:::.s1_stats_reset()
        st <- singlet_study(GSE, base = url)
        expect_equal(singlet:::.s1_stats()$requests, 1L)  # the manifest, nothing else
        sce <- suppressWarnings(singlet:::.s1_read_study(st, c(G1, G3)))
        s <- singlet:::.s1_stats()
        # manifest + pack index + 2 vocabulary packs + one coalesced range per sample
        expect_equal(s$requests, 1L + 1L + 2L + 2L)
        expect_equal(ncol(sce), 65L)
        x1 <- SummarizedExperiment::assay(sce, "counts")[, sce$gsm_id == G1]
        expect_equal(sum(abs(x1 - read_truth(s1_root(), G1))), 0)
        # a second load of the same study asks only for what it does not have: the objects (packs are cached on disk)
        singlet:::.s1_stats_reset()
        singlet:::.s1_clear_memory()
        st2 <- singlet_study(GSE, base = url)
        expect_equal(singlet:::.s1_stats()$requests, 1L)   # a conditional request: 304
        sce2 <- suppressWarnings(singlet:::.s1_read_study(st2, G1))
        expect_equal(singlet:::.s1_stats()$requests, 2L)
        expect_equal(ncol(sce2), 40L)
    })
})
