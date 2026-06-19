# SPDX-License-Identifier: MIT
# Offline tests for the user-facing API plumbing (no network required).

test_that("accession detection recognizes GSE/GSM and rejects paths", {
    expect_true(singlet:::.is_accession("GSE149298"))
    expect_true(singlet:::.is_accession("GSM4495744"))
    expect_false(singlet:::.is_accession("/data/GSE149298.singlet"))
    expect_false(singlet:::.is_accession("not_an_accession"))
    expect_false(singlet:::.is_accession("GSE"))
})

test_that(".gse_of passes GSE through and rejects GSM", {
    expect_identical(singlet:::.gse_of("GSE149298"), "GSE149298")
    expect_error(singlet:::.gse_of("GSM4495744"), "per Series")
})

test_that("cache dir honors SINGLET_CACHE_DIR and explicit arg", {
    tmp <- file.path(tempdir(), "singlet_cache_test")
    on.exit(unlink(tmp, recursive = TRUE), add = TRUE)

    withr_env <- function(value, code) {
        old <- Sys.getenv("SINGLET_CACHE_DIR", unset = NA)
        Sys.setenv(SINGLET_CACHE_DIR = value)
        on.exit({
            if (is.na(old)) Sys.unsetenv("SINGLET_CACHE_DIR")
            else Sys.setenv(SINGLET_CACHE_DIR = old)
        }, add = TRUE)
        force(code)
    }

    d <- withr_env(tmp, singlet:::.singlet_cache_dir())
    expect_true(dir.exists(d))
    expect_identical(normalizePath(d), normalizePath(tmp))

    # explicit cache_dir argument wins
    tmp2 <- file.path(tempdir(), "singlet_cache_test2")
    on.exit(unlink(tmp2, recursive = TRUE), add = TRUE)
    d2 <- singlet:::.singlet_cache_dir(tmp2)
    expect_true(dir.exists(d2))
})

test_that("data base and api base honor environment overrides", {
    old_db <- Sys.getenv("SINGLET_DATA_BASE", unset = NA)
    old_api <- Sys.getenv("SINGLET_API_BASE", unset = NA)
    on.exit({
        if (is.na(old_db)) Sys.unsetenv("SINGLET_DATA_BASE")
        else Sys.setenv(SINGLET_DATA_BASE = old_db)
        if (is.na(old_api)) Sys.unsetenv("SINGLET_API_BASE")
        else Sys.setenv(SINGLET_API_BASE = old_api)
    }, add = TRUE)

    Sys.setenv(SINGLET_DATA_BASE = "https://example.test/d")
    Sys.setenv(SINGLET_API_BASE = "https://example.test/api")
    expect_identical(singlet:::.singlet_data_base(), "https://example.test/d")
    expect_identical(singlet:::.singlet_api_base(), "https://example.test/api")

    Sys.unsetenv("SINGLET_DATA_BASE")
    Sys.unsetenv("SINGLET_API_BASE")
    expect_match(singlet:::.singlet_data_base(), "^https://")
    expect_match(singlet:::.singlet_api_base(), "^https://")
})

test_that("load errors on a non-accession, non-existent input", {
    expect_error(load("definitely_not_a_file_or_accession"),
                 "neither a GSE/GSM accession nor an existing file")
})

test_that("load errors on empty input", {
    expect_error(load(character(0)), "at least one")
})

test_that("find rejects an empty query", {
    expect_error(find(""), "non-empty")
})
