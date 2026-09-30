# SPDX-License-Identifier: MIT
# The singlet_*() aliases and GSM -> parent GSE resolution.
#
# Resolution is tested offline by mocking the HTTP layer; one test hits the
# live API and is skipped on CRAN and when offline.

test_that("the singlet_* aliases are exported", {
    exports <- getNamespaceExports("singlet")
    expect_true(all(c("singlet_load", "singlet_find", "singlet_find_load",
                      "singlet_download") %in% exports))
})

test_that("each alias is the same function as its short name", {
    expect_identical(singlet::singlet_load, singlet::load)
    expect_identical(singlet::singlet_find, singlet::find)
    expect_identical(singlet::singlet_find_load, singlet::find_load)
    expect_identical(singlet::singlet_download, singlet::download)
})

test_that("the aliases validate input like the originals", {
    expect_error(singlet_load(character(0)), "at least one")
    expect_error(singlet_find(""), "non-empty")
    expect_error(singlet_download("definitely_not_an_accession"),
                 "not a GEO accession")
})

test_that("the parent series is read from either API response shape", {
    detail <- list(sample = list(gsm_id = "GSM1", gse_id = "GSE10"),
                   series = list(id = "GSE10"))
    expect_identical(singlet:::.gse_from_payload(detail, "GSM1"), "GSE10")

    listing <- list(data = list(
        list(gsm_id = "GSM2", gse_id = "GSE20"),
        list(gsm_id = "GSM1", gse_id = "GSE10")
    ))
    expect_identical(singlet:::.gse_from_payload(listing, "GSM1"), "GSE10")

    # A listing row for a different sample is not trusted.
    other <- list(data = list(list(gsm_id = "GSM2", gse_id = "GSE20")))
    expect_null(singlet:::.gse_from_payload(other, "GSM1"))
    expect_null(singlet:::.gse_from_payload(NULL, "GSM1"))
    expect_null(singlet:::.gse_from_payload(list(error = "not found"), "GSM1"))
})

test_that("a GSM resolves through the sample-detail endpoint and is cached", {
    seen <- character(0)
    local_mocked_bindings(
        .singlet_http_get = function(url, ...) {
            seen <<- c(seen, url)
            paste0('{"sample": {"gsm_id": "GSM900000001", ',
                   '"gse_id": "GSE900001"}, "series": {"id": "GSE900001"}}')
        },
        .package = "singlet"
    )
    expect_message(singlet:::.gse_of("GSM900000001"), "GSE900001")
    expect_match(seen[[1]], "/gsm/GSM900000001$")

    n_requests <- length(seen)
    expect_identical(singlet:::.gse_of("GSM900000001"), "GSE900001")
    expect_length(seen, n_requests)
})

test_that("GSM resolution falls back to the listing endpoint", {
    local_mocked_bindings(
        .singlet_http_get = function(url, ...) {
            if (grepl("/gsm/", url, fixed = TRUE)) {
                stop("sample lookup failed: HTTP 500")
            }
            paste0('{"total": 2, "data": [',
                   '{"gsm_id": "GSM900000009", "gse_id": "GSE111"}, ',
                   '{"gsm_id": "GSM900000002", "gse_id": "GSE900002"}]}')
        },
        .package = "singlet"
    )
    expect_identical(suppressMessages(singlet:::.gse_of("GSM900000002")),
                     "GSE900002")
})

test_that("an unresolvable GSM fails with the reason and a way out", {
    local_mocked_bindings(
        .singlet_http_get = function(url, ...) {
            stop("sample lookup failed: HTTP 404")
        },
        .package = "singlet"
    )
    err <- tryCatch(singlet:::.gse_of("GSM900000003"),
                    error = conditionMessage)
    expect_match(err, "could not resolve sample GSM900000003", fixed = TRUE)
    expect_match(err, "HTTP 404", fixed = TRUE)
    expect_match(err, "Pass the parent GSE accession", fixed = TRUE)
})

test_that("GSM4120733 resolves to GSE138867 against the live API", {
    skip_on_cran()
    skip_if_offline("singlet.bio")
    cache <- singlet:::.singlet_gsm_parents
    rm(list = ls(cache, all.names = TRUE), envir = cache)
    expect_identical(suppressMessages(singlet:::.gse_of("GSM4120733")),
                     "GSE138867")
})
