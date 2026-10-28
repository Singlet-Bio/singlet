# SPDX-License-Identifier: MIT
# write_s1() / read_s1(): a count matrix as a .s1 sample object of your own, and load() of such a file.
#
# fixtures/s1_write_fixture.s1 is the object singlet.write_s1() (Python) writes for fixture_matrix() below; the Python suite
# (tests/python/test_s1_write.py) builds the same matrix. Both compare what does not depend on the zstd library the codec
# links: the container (sample id, member, kind, digest, offset), the pz2 header and the decoded counts and names. The names
# and tables sections are zstd level 19, whose bytes can differ between zstd releases; built against one zstd, the two
# packages write identical files.

fixture_matrix <- function() {
    g <- 0:39
    cc <- 0:29
    v <- outer(g, cc, function(g, c) ifelse((g * 3 + c * 5) %% 4 == 0, (g * 7 + c * 13) %% 11, 0))
    v[6, 8] <- 70000
    m <- methods::as(Matrix::Matrix(v, sparse = TRUE), "CsparseMatrix")
    dimnames(m) <- list(sprintf("g%02d", 1:40), sprintf("c%02d", 1:30))
    m
}

random_counts <- function(n_genes, n_cells, density, seed = 1) {
    set.seed(seed)
    m <- Matrix::rsparsematrix(n_genes, n_cells, density, rand.x = function(n) as.numeric(sample(c(1:5, 1:5, 1:40, 900), n, TRUE)))
    dimnames(m) <- list(sprintf("ENSG%011d", seq_len(n_genes)), sprintf("CELL%07d", seq_len(n_cells)))
    m
}

test_that("write_s1 and read_s1 round-trip a dgCMatrix exactly", {
    m <- random_counts(300, 500, 0.05)
    f <- tempfile(fileext = ".s1")
    write_s1(m, f)
    r <- read_s1(f)
    expect_s4_class(r, "dgCMatrix")
    expect_identical(dimnames(r), dimnames(m))
    expect_identical(r@i, m@i)
    expect_identical(r@p, m@p)
    expect_identical(r@x, m@x)
    expect_false(file.exists(paste0(f, ".part")))
})

fingerprint <- function(path) {
    info <- s1_members_(.s1_read_head(path))
    con <- file(path, "rb")
    on.exit(close(con))
    seek(con, info$off[[1L]])
    h <- pz2_header_(readBin(con, "raw", n = min(256, info$len[[1L]])))
    keep <- c("m", "n", "nnz", "codec", "value_type", "row_order", "chunk_cols", "n_chunks", "row_names_kind",
              "row_names_id", "col_names_kind", "col_names_id", "digest")
    list(sample_id = info$sample_id, name = info$name, kind = info$kind, off = info$off, digest = info$digest, header = h[keep])
}

test_that("write_s1 writes what singlet.write_s1 writes in Python (shared fixture)", {
    fix <- testthat::test_path("fixtures", "s1_write_fixture.s1")
    skip_if_not(file.exists(fix), "fixture missing")
    f <- tempfile(fileext = ".s1")
    write_s1(fixture_matrix(), f)
    expect_identical(fingerprint(f), fingerprint(fix))
    expect_identical(as.matrix(read_s1(f)), as.matrix(fixture_matrix()))
    expect_identical(as.matrix(read_s1(fix)), as.matrix(fixture_matrix()))
})

test_that("threads do not change the bytes, dense input equals sparse", {
    m <- random_counts(400, 2000, 0.05, seed = 2)
    a <- tempfile(fileext = ".s1")
    b <- tempfile(fileext = ".s1")
    d <- tempfile(fileext = ".s1")
    write_s1(m, a, threads = 1L)
    write_s1(m, b, threads = 8L)
    write_s1(as.matrix(m), d)
    expect_identical(readBin(a, "raw", file.size(a)), readBin(b, "raw", file.size(b)))
    expect_identical(readBin(a, "raw", file.size(a)), readBin(d, "raw", file.size(d)))
})

test_that("write_s1 rejects non-counts and unnamed matrices", {
    m <- random_counts(20, 10, 0.3)
    bad <- m
    bad@x[1] <- 0.5
    expect_error(write_s1(bad, tempfile(fileext = ".s1")), "integers")
    neg <- m
    neg@x[1] <- -2
    expect_error(write_s1(neg, tempfile(fileext = ".s1")), "integers")
    un <- m
    dimnames(un) <- NULL
    expect_error(write_s1(un, tempfile(fileext = ".s1")), "row names")
})

test_that("a gene pack reference needs genes = to read", {
    m <- random_counts(30, 20, 0.3, seed = 3)
    f <- tempfile(fileext = ".s1")
    write_s1(m, f, gene_pack = "0123456789abcdef")
    expect_error(read_s1(f), "vocabulary pack")
    expect_identical(as.matrix(read_s1(f, genes = rownames(m))), as.matrix(m))
})

test_that("SingleCellExperiment in, SingleCellExperiment out, and load() reads the file", {
    skip_if_not_installed("SingleCellExperiment")
    m <- random_counts(50, 40, 0.2, seed = 4)
    sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = m))
    f <- tempfile(fileext = ".s1")
    write_s1(sce, f)
    back <- read_s1(f, as = "sce")
    expect_identical(as.matrix(SummarizedExperiment::assay(back, "counts")), as.matrix(m))
    loaded <- load(f)
    expect_identical(as.matrix(SummarizedExperiment::assay(loaded, "counts")), as.matrix(m))
})
