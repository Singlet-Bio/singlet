# SPDX-License-Identifier: MIT
# Downloaded bundles are verified before they reach the cache. The fixtures are
# written by scripts/make_integrity_fixtures.py.

fx <- function(name) testthat::test_path("fixtures", name)

test_that("a complete bundle passes and yields its manifest", {
    m <- singlet:::.singlet_verify_bundle(fx("good.singlet"))
    expect_identical(m$gse_id, "GSE1")
    expect_identical(m$schema_version, "1.0")
})

test_that("a truncated file is rejected (no central directory)", {
    expect_error(singlet:::.singlet_verify_bundle(fx("truncated.singlet")),
                 "not a complete .singlet archive", class = "singlet_integrity_error")
})

test_that("a member that does not match its checksum is rejected", {
    skip_if_not(singlet:::.singlet_can_hash(), "no SHA-256 available")
    expect_error(singlet:::.singlet_verify_bundle(fx("bad_checksum.singlet")),
                 "does not match its checksum", class = "singlet_integrity_error")
})

test_that("hashing can be skipped for the cheap cache-hit check", {
    expect_no_error(singlet:::.singlet_verify_bundle(fx("bad_checksum.singlet"), hashes = FALSE))
    expect_error(singlet:::.singlet_verify_bundle(fx("truncated.singlet"), hashes = FALSE),
                 class = "singlet_integrity_error")
})

test_that("a newer schema is refused with an update hint", {
    expect_error(singlet:::.singlet_verify_bundle(fx("newer_schema.singlet")),
                 "Update the singlet package", class = "singlet_integrity_error")
})

test_that("a size different from the announced one is rejected", {
    expect_error(singlet:::.singlet_verify_bundle(fx("good.singlet"), expected_size = 10),
                 "cut short or altered", class = "singlet_integrity_error")
    expect_no_error(singlet:::.singlet_verify_bundle(fx("good.singlet"),
                                                     expected_size = file.size(fx("good.singlet"))))
})

test_that("member names that could escape a directory are unsafe", {
    for (bad in c("../evil", "/etc/passwd", "a/../../b", "a\\b", "")) {
        expect_false(singlet:::.singlet_safe_member_name(bad), info = bad)
    }
    expect_true(singlet:::.singlet_safe_member_name("samples/GSM1/exon_counts.1pz"))
})

test_that("a cached file that is no longer a complete archive is replaced, an intact one reused", {
    cache <- withr::local_tempdir()
    withr::local_envvar(SINGLET_CACHE_DIR = cache)
    dest <- file.path(cache, "GSE1.singlet")

    file.copy(fx("good.singlet"), dest)
    calls <- 0L
    local_mocked_bindings(
        .singlet_announced_size = function(url) NA_real_,
        .singlet_bundle_url = function(gse) paste0("file:///", normalizePath(fx("good.singlet"), winslash = "/"))
    )
    expect_identical(singlet:::.singlet_fetch_bundle("GSE1", cache_dir = cache), dest)

    file.copy(fx("truncated.singlet"), dest, overwrite = TRUE)
    out <- singlet:::.singlet_fetch_bundle("GSE1", cache_dir = cache)
    expect_identical(out, dest)
    expect_no_error(singlet:::.singlet_verify_bundle(dest))
})

test_that("a rejected download leaves nothing in the cache", {
    cache <- withr::local_tempdir()
    local_mocked_bindings(
        .singlet_announced_size = function(url) NA_real_,
        .singlet_bundle_url = function(gse) paste0("file:///", normalizePath(fx("truncated.singlet"), winslash = "/"))
    )
    expect_error(singlet:::.singlet_fetch_bundle("GSE1", cache_dir = cache), "rejected and not cached")
    expect_length(list.files(cache, all.files = TRUE, no.. = TRUE), 0L)
})

test_that("accessions that are not accessions are refused before any request", {
    local_mocked_bindings(.singlet_bundle_url = function(gse) stop("no request may be made"))
    for (bad in c("../../etc/passwd", "GSE1/../x", "GSE", "gse138867", "GSE12a", "GSE1;rm")) {
        expect_error(singlet:::.singlet_fetch_bundle(bad, cache_dir = withr::local_tempdir()),
                     "not a GEO accession", info = bad)
    }
})

test_that("a file in the archive that the manifest does not list is rejected", {
    expect_error(singlet:::.singlet_verify_bundle(fx("unlisted_member.singlet")),
                 "not listed in manifest.json", class = "singlet_integrity_error")
})

test_that("a manifest whose checksums are a list, not a name -> hash map, is rejected", {
    expect_error(singlet:::.singlet_verify_bundle(fx("checksums_as_list.singlet")),
                 "map each member name", class = "singlet_integrity_error")
})
