# SPDX-License-Identifier: MIT
# Integrity checks for downloaded .singlet bundles (the R twin of
# python/singlet/_integrity.py). A download only reaches the cache after:
#   1. the byte count matches what the server announced;
#   2. the archive's central directory reads (a truncated file has none);
#   3. manifest.json declares a schema major this version reads;
#   4. every member listed in manifest.json is present and matches its SHA-256.

# Schema major version this reader understands.
.singlet_supported_schema_major <- 1L

.singlet_integrity_error <- function(msg) {
    structure(
        class = c("singlet_integrity_error", "error", "condition"),
        list(message = msg, call = NULL)
    )
}

.singlet_safe_member_name <- function(name) {
    is.character(name) && length(name) == 1L && nzchar(name) &&
        !grepl("^[/\\\\]", name) && !grepl("\\\\", name) &&
        !grepl("[[:cntrl:]]", name) &&
        !any(strsplit(name, "/", fixed = TRUE)[[1]] == "..")
}

# Bytes the server says the file has, or NA when it does not say (or the check
# cannot be made). Uses a HEAD-style header read, not a second download.
.singlet_announced_size <- function(url) {
    hdrs <- tryCatch(curlGetHeaders(url, redirect = TRUE, verify = TRUE),
                     error = function(e) character())
    if (!length(hdrs)) {
        return(NA_real_)
    }
    # curlGetHeaders returns the headers of every hop; the last block is the file.
    starts <- grep("^HTTP/", hdrs)
    block <- if (length(starts)) hdrs[starts[length(starts)]:length(hdrs)] else hdrs
    m <- grep("^content-length:", block, ignore.case = TRUE, value = TRUE)
    if (!length(m)) {
        return(NA_real_)
    }
    suppressWarnings(as.numeric(trimws(sub("^[^:]*:", "", m[length(m)]))))
}

# A SHA-256 can always be computed: tools::sha256sum() on R >= 4.5 (looked up at run time, so older R has no reference to a
# missing object), else the digest package, else the package's own compiled SHA-256 over the file's bytes.
.singlet_can_hash <- function() TRUE

.singlet_sha256 <- function(path) {
    f <- get0("sha256sum", envir = asNamespace("tools"), mode = "function", inherits = FALSE)
    if (!is.null(f)) {
        return(unname(f(path)))
    }
    if (requireNamespace("digest", quietly = TRUE)) {
        return(digest::digest(file = path, algo = "sha256"))
    }
    pz2_sha256_(readBin(path, "raw", n = file.size(path)))
}

# Check a bundle on disk and return its parsed manifest; raises a
# `singlet_integrity_error` describing what is wrong. `hashes = FALSE` checks
# only size, structure and schema (what a cache hit costs).
.singlet_verify_bundle <- function(path, expected_size = NA_real_, hashes = TRUE) {
    name <- basename(path)
    size <- suppressWarnings(file.size(path))
    if (is.na(size) || size == 0) {
        stop(.singlet_integrity_error(sprintf("%s is empty or unreadable", name)))
    }
    if (!is.na(expected_size) && size != expected_size) {
        stop(.singlet_integrity_error(sprintf(
            "%s is %s bytes but the server announced %s: the download was cut short or altered",
            name, format(size, big.mark = ","), format(expected_size, big.mark = ","))))
    }
    members <- tryCatch(
        suppressWarnings(utils::unzip(path, list = TRUE)),
        error = function(e) NULL
    )
    if (is.null(members) || !nrow(members)) {
        stop(.singlet_integrity_error(sprintf("%s is not a complete .singlet archive", name)))
    }
    names_in_zip <- members$Name
    if (!"manifest.json" %in% names_in_zip) {
        stop(.singlet_integrity_error("not a .singlet bundle: manifest.json is missing"))
    }
    if (members$Length[match("manifest.json", members$Name)] > 16 * 1024^2) {
        stop(.singlet_integrity_error("manifest.json is implausibly large"))
    }
    manifest <- tryCatch({
        # extracted, not read through unz(): before R 4.5 unz() returns nothing for these zip64 archives
        mdir <- tempfile("singlet_manifest_")
        on.exit(unlink(mdir, recursive = TRUE), add = TRUE)
        mf <- utils::unzip(path, files = "manifest.json", exdir = mdir)
        jsonlite::fromJSON(paste(readLines(mf, warn = FALSE, encoding = "UTF-8"), collapse = "\n"),
                           simplifyVector = FALSE)
    }, error = function(e) {
        stop(.singlet_integrity_error("manifest.json could not be read"))
    })
    if (!is.list(manifest)) {
        stop(.singlet_integrity_error("manifest.json must hold a JSON object"))
    }
    major <- suppressWarnings(as.integer(sub("^([0-9]+).*$", "\\1", as.character(if (is.null(manifest$schema_version)) "" else manifest$schema_version))))
    if (length(major) != 1L || is.na(major)) {
        stop(.singlet_integrity_error("manifest.json has no valid schema_version"))
    }
    if (major != .singlet_supported_schema_major) {
        stop(.singlet_integrity_error(sprintf(
            "this bundle uses schema %s, but this version of singlet reads schema %d.x. Update the singlet package to read it.",
            manifest$schema_version, .singlet_supported_schema_major)))
    }
    checksums <- manifest$checksums
    if (!is.list(checksums) || !length(checksums) || is.null(names(checksums)) || !all(nzchar(names(checksums)))) {
        stop(.singlet_integrity_error("manifest.json must map each member name to its checksum"))
    }
    if (anyDuplicated(names_in_zip)) {
        stop(.singlet_integrity_error("the archive holds two members with the same name"))
    }
    # Every file in the archive must be vouched for by the manifest.
    unlisted <- setdiff(names_in_zip[!grepl("/$", names_in_zip)], c("manifest.json", names(checksums)))
    if (length(unlisted)) {
        stop(.singlet_integrity_error(sprintf("%s is in the archive but not listed in manifest.json", unlisted[[1]])))
    }
    for (n in names_in_zip) {
        if (!.singlet_safe_member_name(n)) {
            stop(.singlet_integrity_error(sprintf("unsafe member name in archive: %s", n)))
        }
    }
    for (n in names(checksums)) {
        if (!.singlet_safe_member_name(n)) {
            stop(.singlet_integrity_error(sprintf("unsafe member name in manifest: %s", n)))
        }
        if (!n %in% names_in_zip) {
            stop(.singlet_integrity_error(sprintf("%s is listed in manifest.json but missing from the archive", n)))
        }
        if (!grepl("^[0-9a-fA-F]{64}$", as.character(checksums[[n]]))) {
            stop(.singlet_integrity_error(sprintf("manifest.json has a malformed checksum for %s", n)))
        }
    }
    if (hashes && !.singlet_can_hash()) {
        warning("cannot verify checksums: R >= 4.5 or the 'digest' package is needed; checked size and structure only",
                call. = FALSE)
        hashes <- FALSE
    }
    if (hashes) {
        scratch <- tempfile("singlet-verify-")
        dir.create(scratch)
        on.exit(unlink(scratch, recursive = TRUE), add = TRUE)
        for (n in names(checksums)) {
            got <- tryCatch({
                utils::unzip(path, files = n, exdir = scratch, junkpaths = TRUE)
                f <- file.path(scratch, basename(n))
                on.exit(unlink(f), add = TRUE)
                h <- .singlet_sha256(f)
                unlink(f)
                h
            }, error = function(e) NA_character_,
               warning = function(w) NA_character_)
            if (is.na(got)) {
                stop(.singlet_integrity_error(sprintf("%s is corrupt", n)))
            }
            if (!identical(tolower(got), tolower(as.character(checksums[[n]])))) {
                stop(.singlet_integrity_error(sprintf("%s does not match its checksum in manifest.json", n)))
            }
        }
    }
    manifest
}
