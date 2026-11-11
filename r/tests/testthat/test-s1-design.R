# SPDX-License-Identifier: MIT
# s1_design(), s1_contrasts(), s1_sample_status() on the local fixture sidecars (tests/data/design/s1/d, shared with the
# Python tests) and a fake transport. No network.

fx_dir <- function() {
    for (p in c(testthat::test_path("..", "..", "..", "tests", "data", "design"), testthat::test_path("fixtures", "design"))) {
        if (dir.exists(file.path(p, "s1", "d"))) return(normalizePath(p, winslash = "/"))
    }
    NA_character_
}
skip_if_no_design_fixture <- function() if (is.na(fx_dir())) testthat::skip("design fixture sidecars are not available")
fx_doc <- function(gse) jsonlite::fromJSON(file.path(fx_dir(), "s1", "d", paste0(gse, ".design.json")), simplifyVector = FALSE)
tmp_cache <- function() {
    d <- tempfile("s1design-cache-")
    dir.create(d)
    d
}

# A fake remote data host: replaces the package's GET with an in-memory one that answers with ETags and 304.
fake_host <- function(docs, etag = "v1", fail = NULL, env = parent.frame()) {
    st <- new.env()
    st$requests <- list()
    st$docs <- docs
    st$etag <- etag
    st$fail <- fail
    testthat::local_mocked_bindings(
        .s1_transport = function(base = NULL) list(kind = "http", base = "https://fake.invalid"),
        .s1_get = function(tr, key, range = NULL, expect_total = NA_real_, if_none_match = NULL, retries = 3L) {
            st$requests[[length(st$requests) + 1L]] <- list(key = key, inm = if_none_match)
            if (!is.null(st$fail)) singlet:::.s1_stop(st$fail)
            gse <- sub("\\..*$", "", basename(key))
            if (is.null(st$docs[[gse]])) singlet:::.s1_stop(paste(key, "HTTP 404"), "singlet_s1_not_found")
            if (identical(if_none_match, st$etag)) return(list(status = 304L, body = raw(0), etag = st$etag, total = NA_real_))
            body <- charToRaw(jsonlite::toJSON(st$docs[[gse]], auto_unbox = TRUE, null = "null"))
            list(status = 200L, body = body, etag = st$etag, total = length(body))
        },
        .package = "singlet", .env = env)
    st
}

test_that("s1_design returns the tables, contrasts, tier and provenance from a local directory", {
    skip_if_no_design_fixture()
    d <- s1_design("gse190760", base = fx_dir())
    expect_s3_class(d, "singlet_design")
    expect_identical(d$gse, "GSE190760")
    expect_identical(d$tier, "candidate")
    expect_identical(d$source, "local")
    expect_identical(d$factors$factor_id, c("F1", "F2"))
    expect_identical(d$factors$type, c("response", "time"))
    expect_identical(d$factors$n_levels, c(2L, 4L))
    expect_equal(nrow(d$levels), 6L)
    expect_setequal(d$levels$role, c("other", "reference", "test"))
    expect_equal(nrow(d$units), 6L)
    expect_true(all(c("F1", "F2", "donor", "pool_type", "status", "cov_sex") %in% names(d$units)))
    expect_identical(d$units$F2[1:2], c("F2.a", "F2.b"))
    expect_equal(length(d$contrasts), 4L)
    expect_identical(d$provenance$method, "claude-code-subagent")
    for (fr in list(d$factors, d$levels, d$units)) expect_identical(attr(fr, "tier"), "candidate")
})

test_that("printing repeats the caveat that candidate means model-curated and unverified", {
    skip_if_no_design_fixture()
    d <- s1_design("GSE190760", base = fx_dir())
    expect_output(print(d), "model-curated and unverified")
    expect_output(print(s1_contrasts("GSE190760", base = fx_dir())), "model-curated and unverified")
    expect_match(attr(s1_sample_status("GSE190760", base = fx_dir()), "caveat"), "model-curated and unverified")
    doc <- fx_doc("GSE190760")
    doc$tier <- "confirmed"
    fake_host(list(GSE190760 = doc))
    out <- capture.output(print(s1_design("GSE190760", cache_dir = tmp_cache())))
    expect_false(any(grepl("model-curated and unverified", out)))
    expect_true(any(grepl("confirmed", out)))
})

test_that("a study without a sidecar is a clear error of its own class", {
    skip_if_no_design_fixture()
    e <- tryCatch(s1_design("GSE1", base = fx_dir()), error = function(e) e)
    expect_s3_class(e, "singlet_design_not_found")
    expect_s3_class(e, "singlet_s1_not_found")
    expect_match(conditionMessage(e), "GSE1 has no design sidecar")
    expect_match(conditionMessage(e), "s1/d/GSE1.design.json", fixed = TRUE)
    expect_error(s1_design("GSM12", base = fx_dir()), "not a GEO Series accession")
    fake_host(list())
    expect_error(s1_design("GSE2", cache_dir = tmp_cache()), class = "singlet_design_not_found")
})

test_that("a sidecar of another kind or a newer schema is refused", {
    skip_if_no_design_fixture()
    doc <- fx_doc("GSE190760")
    newer <- doc
    newer$schema <- "singlet-design-catalog-1.0"
    fake_host(list(GSE190760 = newer))
    expect_error(s1_design("GSE190760", cache_dir = tmp_cache()), "newer")
    fake_host(list(GSE190760 = list(hello = 1)))
    expect_error(s1_design("GSE190760", cache_dir = tmp_cache()), "not a singlet design")
    wrong <- doc
    wrong$study <- "GSE5"
    fake_host(list(GSE190760 = wrong))
    expect_error(s1_design("GSE190760", cache_dir = tmp_cache()), "describes GSE5")
})

test_that("the sidecar is cached with its ETag and used offline", {
    skip_if_no_design_fixture()
    doc <- fx_doc("GSE190760")
    cache <- tmp_cache()
    h <- fake_host(list(GSE190760 = doc))
    d1 <- s1_design("GSE190760", cache_dir = cache)
    expect_identical(d1$source, "network")
    expect_null(h$requests[[1L]]$inm)
    d2 <- s1_design("GSE190760", cache_dir = cache)
    expect_identical(d2$source, "cache-304")
    expect_identical(h$requests[[2L]]$inm, "v1")
    expect_identical(d2$units, d1$units)
    h$fail <- "connection reset"
    expect_warning(d3 <- s1_design("GSE190760", cache_dir = cache), "cached sidecar")
    expect_identical(d3$source, "cache-offline")
    h$fail <- NULL
    doc$tier <- "confirmed"
    h$docs$GSE190760 <- doc
    h$etag <- "v2"
    expect_identical(s1_design("GSE190760", cache_dir = cache)$tier, "confirmed")
})

test_that("s1_contrasts filters by factor type, grade and tier", {
    skip_if_no_design_fixture()
    studies <- c("GSE190760", "GSE190487")
    b <- fx_dir()
    all <- s1_contrasts(studies, base = b)
    expect_equal(length(all), 5L)
    expect_true(all(vapply(all, function(c) !is.null(c$study) && !is.null(c$tier), NA)))
    t <- s1_contrasts(studies, factor_type = "time", base = b)
    expect_identical(vapply(t, function(c) c$id, ""), paste0("GSE190760.c", 2:4))
    expect_equal(length(s1_contrasts(studies, factor_type = c("time", "cell_fraction"), base = b)), 4L)
    a <- s1_contrasts(studies, min_grade = "A", base = b)
    expect_equal(length(a), 3L)
    expect_setequal(vapply(a, function(c) c$grade, ""), "A")
    expect_setequal(vapply(s1_contrasts(studies, min_grade = "b", base = b), function(c) c$grade, ""), c("A", "B"))
    expect_equal(length(s1_contrasts(studies, min_grade = "D", base = b)), 5L)
    expect_identical(vapply(s1_contrasts(studies, tier = "unresolved", base = b), function(c) c$study, ""), "GSE190487")
    expect_equal(length(s1_contrasts(studies, tier = "confirmed", base = b)), 0L)
    combo <- as.data.frame(s1_contrasts(studies, factor_type = "time", min_grade = "A", tier = "candidate", base = b))
    expect_equal(nrow(combo), 3L)
    expect_setequal(combo$tier, "candidate")
    expect_match(attr(combo, "caveat"), "model-curated and unverified")
})

test_that("s1_contrasts checks its arguments and handles studies without a sidecar", {
    skip_if_no_design_fixture()
    b <- fx_dir()
    expect_error(s1_contrasts("GSE190760", min_grade = "E", base = b), "min_grade")
    expect_error(s1_contrasts("GSE190760", tier = "verified", base = b), "unknown tier")
    expect_warning(r <- s1_contrasts(c("GSE7", "GSE190487"), base = b), "GSE7")
    expect_equal(length(r), 1L)
    expect_equal(length(s1_contrasts(c("GSE7", "GSE190487"), base = b, missing = "skip")), 1L)
    expect_error(s1_contrasts("GSE7", base = b, missing = "error"), class = "singlet_design_not_found")
})

test_that("s1_sample_status gives the status of each unit", {
    skip_if_no_design_fixture()
    st <- s1_sample_status("GSE190760", base = fx_dir())
    expect_equal(nrow(st), 6L)
    expect_identical(st$gsm[1:2], c("GSM5731055", "GSM5731056"))
    expect_setequal(st$status, "scheduled")
    expect_setequal(st$tier, "candidate")
    expect_identical(attr(st, "tier"), "candidate")
    doc <- fx_doc("GSE190760")
    doc$units[[1]]$status <- "held"
    doc$units[[2]]$status <- "counts_available"
    fake_host(list(GSE190760 = doc))
    st2 <- s1_sample_status("GSE190760", cache_dir = tmp_cache())
    expect_identical(st2$status[1:3], c("held", "counts_available", "scheduled"))
})
