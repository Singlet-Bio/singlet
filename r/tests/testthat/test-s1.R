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
# e.g. tests/python/_s1_range_server.py) enables the HTTP tests; the io_threads tests start that server themselves.

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
        out <- file.path(tempdir(), "s1_synth")
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
        expect_error(singlet_study("GSE1", base = root), "not in the atlas")
        expect_error(singlet_study("GSE1", base = root), "not in the atlas")
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
        # with the fallback off (the default since the retirement), a study without a manifest is said not to be in the atlas
        withr::with_envvar(c(SINGLET_LEGACY_FALLBACK = "0"), expect_error(download("GSE1"), "not in the atlas"))
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

test_that("load() and singlet_cells() carry the study API's sample facts as factors", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    k562 <- list(sample_kind = "cell_line", tissue_type = "cell line", cell_line = list(list(id = "CVCL_0004", name = "K-562")))
    adj <- list(tissue_type = "tissue", tumor_sample_type = "adjacent normal",
                disease = list(list(id = "MONDO:0005061", label = "lung adenocarcinoma")))
    testthat::local_mocked_bindings(.sample_facts_for_study = function(study) structure(list(k562, adj), names = c(G1, G3)),
                                    .package = "singlet")
    with_s1_cache({
        sce <- suppressWarnings(singlet:::.s1_read_study(singlet_study(GSE, base = root), c(G1, G3)))
        cd <- SummarizedExperiment::colData(sce)
        expect_true(all(singlet:::.sample_fact_columns %in% colnames(cd)))
        expect_true(is.factor(cd$tissue_type) && is.factor(cd$cell_line_cellosaurus_id))
        expect_identical(unique(as.character(cd$cell_line[cd$gsm_id == G1])), "K-562")
        expect_true(all(is.na(cd$disease[cd$gsm_id == G1])))
        expect_identical(unique(as.character(cd$disease_ontology_term_id[cd$gsm_id == G3])), "MONDO:0005061")
        sub <- singlet_cells(singlet_study(GSE, base = root), called, samples = G1)
        expect_identical(unique(as.character(SummarizedExperiment::colData(sub)$tissue_type)), "cell line")
        off <- suppressWarnings(singlet:::.s1_read_study(singlet_study(GSE, base = root), G1, sample_facts = FALSE))
        expect_false(any(singlet:::.sample_fact_columns %in% colnames(SummarizedExperiment::colData(off))))
    })
})


# ---------------------------------------------------------------------------
# Concurrent range requests (io_threads) against tests/python/_s1_range_server.py
# ---------------------------------------------------------------------------

# A range server of `root` in another process (Python), stopped when the calling test ends; returns its base URL. `...` are
# server options: "--range-status", "200" answers ranges with 200 and a Content-Range (as rclone serve s3 does);
# "--ignore-range" sends the whole object; "--delay", "0.05" holds every request (a slow link).
s1_serve <- function(root, ..., env = parent.frame()) {
    skip_if_not_installed("curl")
    py <- Sys.getenv("SINGLET_PYTHON", Sys.which("python"))
    script <- normalizePath(file.path(testthat::test_path("..", "..", ".."), "tests", "python", "_s1_range_server.py"), mustWork = FALSE)
    skip_if(!nzchar(py) || !file.exists(script), "needs Python and tests/python/_s1_range_server.py")
    out <- tempfile(fileext = ".url")
    system2(py, c("-P", shQuote(script), shQuote(root), ..., "--ttl", "600", "--url-file", shQuote(out)), stdout = FALSE, stderr = FALSE,
            wait = FALSE)
    u <- character(0)
    for (i in 1:150) {
        u <- if (file.exists(out)) readLines(out, warn = FALSE) else character(0)
        if (length(u) && grepl("^http://", u[[1L]])) break
        Sys.sleep(0.1)
    }
    skip_if(!length(u) || !grepl("^http://", u[[1L]]), "the range server did not start")
    url <- u[[1L]]
    withr::defer(try(curl::curl_fetch_memory(paste0(url, "/_stop")), silent = TRUE), envir = env)
    url
}

s1_server_stats <- function(url) jsonlite::fromJSON(rawToChar(curl::curl_fetch_memory(paste0(url, "/_stats"))$content))

# The parts of a load that must not depend on how the bytes were fetched.
sce_content <- function(sce) {
    list(cn = colnames(sce), rn = rownames(sce), assays = lapply(SummarizedExperiment::assays(sce), as.matrix),
         cd = as.data.frame(SummarizedExperiment::colData(sce)), skipped = S4Vectors::metadata(sce)$skipped_samples)
}

test_that("io_threads is \"auto\" (8 over HTTP, 1 locally) or a whole number of at least 1", {
    expect_identical(singlet:::.s1_io_threads("auto", list(kind = "http")), 8L)
    expect_identical(singlet:::.s1_io_threads("auto", list(kind = "dir")), 1L)
    expect_identical(singlet:::.s1_io_threads("auto"), 1L)
    expect_identical(singlet:::.s1_io_threads(1), 1L)
    expect_identical(singlet:::.s1_io_threads(16L, list(kind = "http")), 16L)
    for (bad in list(0, -1, 2.5, NA, "8", c(1, 2), NULL)) expect_error(singlet:::.s1_io_threads(bad), "io_threads")
    expect_error(load("GSE1", io_threads = 0), "io_threads")
})

test_that("io_threads 1 and 8 load the same study over HTTP, and 8 has requests in flight together", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    url <- s1_serve(root, "--delay", "0.05")
    with_s1_cache({
        # the packs, once, so that every read below asks for sample bytes only
        suppressWarnings(singlet:::.s1_read_study(singlet_study(GSE, base = url), G1, sample_facts = FALSE))
        res <- list()
        for (io in c(1L, 8L)) {
            for (wm in c(48 * 1024^2, 0)) {  # whole objects, then the chunk path (members' prefixes, then chunk runs)
                singlet:::.s1_clear_memory()
                st <- singlet_study(GSE, base = url, force = TRUE)
                before <- s1_server_stats(url)
                singlet:::.s1_stats_reset()
                sce <- suppressWarnings(singlet:::.s1_read_study(st, c(G1, G2, G3), io_threads = io, whole_max = wm, sample_facts = FALSE))
                after <- s1_server_stats(url)
                res[[sprintf("%d_%.0f", io, wm)]] <- list(sce = sce_content(sce), requests = singlet:::.s1_stats()$requests,
                                                          server = after$requests - before$requests)
            }
        }
        for (wm in c("50331648", "0")) {
            one <- res[[paste0("1_", wm)]]
            eight <- res[[paste0("8_", wm)]]
            expect_identical(eight$sce, one$sce)
            expect_identical(eight$requests, one$requests)  # the same ranges, only sent together
            expect_identical(one$server, one$requests)
        }
        expect_identical(res[["8_0"]]$sce, res[["8_50331648"]]$sce)
        x1 <- res[["8_0"]]$sce$assays$counts[, startsWith(res[["8_0"]]$sce$cn, G1)]
        expect_equal(sum(abs(x1 - as.matrix(read_truth(root, G1)))), 0)
        expect_identical(res[["8_0"]]$sce$skipped$gsm_id, G2)
        # io_threads = 1 never had two requests at the server at once; io_threads = 8 did (the samples' ranges went together)
        srv1 <- s1_serve(root, "--delay", "0.05")
        singlet:::.s1_clear_memory()
        suppressWarnings(singlet:::.s1_read_study(singlet_study(GSE, base = srv1, force = TRUE), c(G1, G3), io_threads = 1L, sample_facts = FALSE))
        expect_identical(s1_server_stats(srv1)$max_inflight, 1L)
        expect_gt(s1_server_stats(url)$max_inflight, 1L)
        # singlet_cells() too
        st <- singlet_study(GSE, base = url, force = TRUE)
        c1 <- singlet_cells(st, n_umi_exon > 0 & called, samples = c(G1, G3), sample_facts = FALSE, io_threads = 1L)
        c8 <- singlet_cells(st, n_umi_exon > 0 & called, samples = c(G1, G3), sample_facts = FALSE, io_threads = 8L)
        expect_identical(sce_content(c8), sce_content(c1))
    })
})

test_that("a 200 answer whose Content-Range names the requested bytes is a range answer", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    u206 <- s1_serve(root)
    u200 <- s1_serve(root, "--range-status", "200")
    with_s1_cache({
        got <- lapply(list(u206, u200), function(u) {
            lapply(c(1L, 8L), function(io) {
                singlet:::.s1_clear_memory()
                sce_content(suppressWarnings(singlet:::.s1_read_study(singlet_study(GSE, base = u, force = TRUE), c(G1, G3),
                                                                      io_threads = io, whole_max = 0, sample_facts = FALSE)))
            })
        })
        expect_identical(got[[2L]][[1L]], got[[1L]][[1L]])
        expect_identical(got[[2L]][[2L]], got[[1L]][[1L]])
        expect_identical(got[[1L]][[2L]], got[[1L]][[1L]])
    })
})

test_that("a server that ignores Range is refused, one request at a time or many", {
    root <- skip_if_no_s1()
    skip_if_not_installed("SingleCellExperiment")
    url <- s1_serve(root, "--ignore-range")
    with_s1_cache({
        for (io in c(1L, 8L)) {
            singlet:::.s1_clear_memory()
            st <- singlet_study(GSE, base = url, force = TRUE)  # the manifest is a whole-object GET: fine
            expect_error(singlet:::.s1_read_study(st, c(G1, G3), io_threads = io, sample_facts = FALSE), "ignored the Range header",
                         class = "singlet_s1_integrity")
        }
    })
})

test_that("read_s1() reads a URL, in pieces when io_threads allows, and refuses a server that ignores Range", {
    fix <- testthat::test_path("fixtures", "s1_write_fixture.s1")
    dir <- normalizePath(dirname(fix), winslash = "/")
    local <- read_s1(fix)
    u206 <- s1_serve(dir, "--delay", "0.02")
    u200 <- s1_serve(dir, "--range-status", "200")
    for (u in c(u206, u200)) {
        expect_identical(read_s1(paste0(u, "/s1_write_fixture.s1"), io_threads = 1L), local)
        expect_identical(read_s1(paste0(u, "/s1_write_fixture.s1")), local)
    }
    # a member longer than the piece size goes out as several ranges in flight together, joined in order
    withr::local_options(singlet.s1_piece_min = 64)
    singlet:::.s1_stats_reset()
    expect_identical(read_s1(paste0(u206, "/s1_write_fixture.s1"), io_threads = 8L), local)
    expect_gt(singlet:::.s1_stats()$requests, 2L + 1L)
    expect_gt(s1_server_stats(u206)$max_inflight, 1L)
    expect_identical(read_s1(paste0(u200, "/s1_write_fixture.s1"), io_threads = 8L), local)
    uig <- s1_serve(dir, "--ignore-range")
    expect_error(read_s1(paste0(uig, "/s1_write_fixture.s1"), io_threads = 1L), "ignored the Range header", class = "singlet_s1_integrity")
    expect_error(read_s1(paste0(uig, "/s1_write_fixture.s1"), io_threads = 8L), "ignored the Range header", class = "singlet_s1_integrity")
})

test_that("download() over HTTP with io_threads 8 mirrors the objects byte for byte", {
    root <- skip_if_no_s1()
    url <- s1_serve(root)
    with_s1_cache({
        withr::local_envvar(SINGLET_DATA_BASE = url)
        withr::local_options(singlet.s1_piece_min = 1024)  # objects of a few KiB still go out in pieces
        mp <- download(GSE, io_threads = 8L)
        st <- singlet_study(GSE, base = root)
        for (g in c(G1, G2, G3)) {
            key <- st$samples[[g]]$key
            a <- readBin(file.path(root, key), "raw", file.size(file.path(root, key)))
            b <- file.path(dirname(dirname(mp)), "s", g, basename(key))
            expect_identical(readBin(b, "raw", file.size(b)), a)
        }
    })
})
