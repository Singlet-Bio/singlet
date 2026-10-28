# SPDX-License-Identifier: MIT
# The .s1 read path (docs/format-s1.md): a study is a manifest (s1/m/<GSE>.json) plus one immutable object per sample
# (s1/s/<GSM>/<digest>.s1). load() fetches only the members it needs with HTTP Range requests, decodes each matrix once
# in C++ straight into a dgCMatrix (already aggregated to genes, restricted to the called cells), and takes names from
# vocabulary packs. Mirrors python/singlet/s1/client.py; see bench/compression/notes/m_w4.md.
#
# Network: the curl package (a Suggests dependency, already used for search) is required for ranged reads from
# the public host; a local directory laid out like the server needs nothing. The data host is
# SINGLET_DATA_BASE (default https://data.singlet.bio; a directory or file:// URL with the server layout also works).


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
.s1_stop <- function(msg, class = NULL) {
    new <- c(class, "singlet_s1_error")
    # DUAL (removed in a later cleanup): the old singlet_v2_* classes stay on every condition, so handlers written for them still catch.
    old <- sub("^singlet_s1_", "singlet_v2_", new)
    stop(structure(class = c(new, old, "error", "condition"),
                   list(message = msg, call = NULL)))
}

# GET s1/ref/<rel>; DUAL: on 404 the same object under the old v2/ref/ prefix.
.s1_get_ref <- function(tr, rel) {
    tryCatch(.s1_get(tr, paste0("s1/ref/", rel)),
             singlet_s1_not_found = function(e) .s1_get(tr, paste0("v2/ref/", rel)))
}

# Transition switch (docs/s1-migration.md): off since the legacy bundles were retired (data.singlet.bio answers 410 for
# them). SINGLET_LEGACY_FALLBACK overrides either way.
.STORE_LEGACY_FALLBACK_DEFAULT <- FALSE
.s1_legacy_fallback <- function() {
    v <- tolower(Sys.getenv("SINGLET_LEGACY_FALLBACK", ""))
    if (v %in% c("1", "true", "yes", "on")) return(TRUE)
    if (v %in% c("0", "false", "no", "off")) return(FALSE)
    .STORE_LEGACY_FALLBACK_DEFAULT
}

.s1_no_study_message <- function(msg) {
    paste0(msg, ". The study is not in the atlas: check the accession, or see https://singlet.bio/known-issues for studies ",
           "Research Alpha withholds (legacy .singlet bundles are retired).")
}


# ---------------------------------------------------------------------------
# Session state: request counters, manifests and packs already read
# ---------------------------------------------------------------------------
.s1_mem <- new.env(parent = emptyenv())
.s1_mem$requests <- 0L
.s1_mem$bytes <- 0
.s1_mem$studies <- list()
.s1_mem$packs <- list()
.s1_mem$vocab <- list()
.s1_mem$pack_index <- list()

# Requests and bytes issued by the .s1 client in this session (for tests and the network model).
.s1_stats <- function() list(requests = .s1_mem$requests, bytes = .s1_mem$bytes)
.s1_stats_reset <- function() {
    .s1_mem$requests <- 0L
    .s1_mem$bytes <- 0
    invisible(NULL)
}
.s1_clear_memory <- function() {
    .s1_mem$studies <- list()
    .s1_mem$packs <- list()
    .s1_mem$vocab <- list()
    .s1_mem$pack_index <- list()
    invisible(NULL)
}
.s1_count <- function(n_bytes) {
    .s1_mem$requests <- .s1_mem$requests + 1L
    .s1_mem$bytes <- .s1_mem$bytes + n_bytes
}

.s1_root <- function(cache_dir = NULL) {
    d <- file.path(.singlet_cache_dir(cache_dir), "s1")  # same layout as the Python client (<cache>/s1)
    if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
    d
}


# ---------------------------------------------------------------------------
# Transport: ranged HTTP GET (curl) or a local directory with the server layout
# ---------------------------------------------------------------------------
.s1_transport <- function(base = NULL) {
    b <- if (is.null(base) || !nzchar(base)) .singlet_data_base() else sub("/data$", "", sub("/+$", "", base))
    if (grepl("^https?://", b)) {
        return(list(kind = "http", base = b))
    }
    root <- sub("^file://", "", b)
    if (grepl("^/[A-Za-z]:", root)) root <- substring(root, 2L)
    list(kind = "dir", base = normalizePath(root, winslash = "/", mustWork = FALSE), root = root)
}

.s1_require_curl <- function() {
    if (!requireNamespace("curl", quietly = TRUE)) {
        stop("the .s1 client needs the 'curl' package for ranged downloads: install.packages(\"curl\")", call. = FALSE)
    }
}

.s1_hdr_range <- function(range) sprintf("bytes=%.0f-%.0f", range[[1L]], range[[2L]])

.s1_handle <- function(range = NULL, if_none_match = NULL) {
    h <- curl::new_handle()
    curl::handle_setopt(h, timeout = 300, connecttimeout = 30, followlocation = TRUE, maxredirs = 3L)
    hdr <- list("User-Agent" = .singlet_user_agent(), "Accept-Encoding" = "identity")
    if (!is.null(range)) hdr[["Range"]] <- .s1_hdr_range(range)
    if (!is.null(if_none_match)) hdr[["If-None-Match"]] <- if_none_match
    do.call(curl::handle_setheaders, c(list(h), hdr))
    h
}

# Check one response against what was asked; returns list(status, body, etag, total).
.s1_check_response <- function(res, url, range, expect_total) {
    status <- res$status_code
    hdr <- curl::parse_headers_list(res$headers)
    etag <- hdr[["etag"]]
    if (status == 304L) {
        .s1_count(0)
        return(list(status = 304L, body = raw(0), etag = etag, total = NA_real_))
    }
    if (status == 404L) .s1_stop(sprintf("%s: HTTP 404", url), "singlet_s1_not_found")
    # 410: a retired object (a legacy .singlet bundle, or a study Research Alpha withholds)
    if (status == 410L) .s1_stop(.s1_no_study_message(sprintf("%s: HTTP 410 (gone)", url)), "singlet_s1_not_found")
    if (status == 403L) {
        .s1_stop(sprintf("%s: HTTP 403 (the data host rejects this client; check SINGLET_DATA_BASE)", url))
    }
    if (status == 416L) .s1_stop(sprintf("%s: HTTP 416, range %s is outside the object", url, .s1_hdr_range(range)), "singlet_s1_integrity")
    if (!status %in% c(200L, 206L)) .s1_stop(sprintf("%s: HTTP %d", url, status))
    body <- res$content
    total <- NA_real_
    if (status == 206L) {
        cr <- hdr[["content-range"]]
        m <- if (is.null(cr)) NULL else regmatches(cr, regexec("^bytes ([0-9]+)-([0-9]+)/([0-9]+|\\*)$", cr))[[1L]]
        if (is.null(range) || length(m) == 0L) {
            .s1_stop(sprintf("%s: missing or unexpected Content-Range %s", url, if (is.null(cr)) "(none)" else cr), "singlet_s1_integrity")
        }
        a <- as.numeric(m[[2L]]); b <- as.numeric(m[[3L]])
        total <- if (m[[4L]] == "*") NA_real_ else as.numeric(m[[4L]])
        if (a != range[[1L]] || b != range[[2L]]) {
            .s1_stop(sprintf("%s: asked bytes %.0f-%.0f, server sent %.0f-%.0f", url, range[[1L]], range[[2L]], a, b), "singlet_s1_integrity")
        }
        if (!is.na(expect_total) && !identical(total, expect_total)) {
            .s1_stop(sprintf("%s: object is %.0f bytes, expected %.0f (stale or replaced object)", url, total, expect_total), "singlet_s1_integrity")
        }
        if (length(body) != b - a + 1) {
            .s1_stop(sprintf("%s: body is %d bytes, expected %.0f", url, length(body), b - a + 1), "singlet_s1_integrity")
        }
    } else if (!is.null(range)) {
        cl <- length(body)
        if (!(range[[1L]] == 0 && range[[2L]] >= cl - 1)) {
            .s1_stop(sprintf("%s: the server ignored the Range header", url), "singlet_s1_integrity")
        }
        total <- cl
    }
    .s1_count(length(body))
    list(status = status, body = body, etag = etag, total = total)
}

.s1_get_dir <- function(tr, key, range, expect_total, if_none_match) {
    p <- file.path(tr$root, key)
    if (!file.exists(p)) .s1_stop(sprintf("%s: not found", p), "singlet_s1_not_found")
    size <- file.size(p)
    if (!is.na(expect_total) && size != expect_total) {
        .s1_stop(sprintf("%s: object is %.0f bytes, expected %.0f", p, size, expect_total), "singlet_s1_integrity")
    }
    etag <- sprintf("\"%.0f-%.0f\"", size, as.numeric(file.mtime(p)) * 1000)
    if (!is.null(if_none_match) && is.null(range) && identical(if_none_match, etag)) {
        .s1_count(0)
        return(list(status = 304L, body = raw(0), etag = etag, total = size))
    }
    a <- if (is.null(range)) 0 else range[[1L]]
    b <- if (is.null(range)) size - 1 else range[[2L]]
    if (a < 0 || b < a || b >= size) .s1_stop(sprintf("%s: range outside the %.0f-byte object", p, size), "singlet_s1_integrity")
    con <- file(p, "rb")
    on.exit(close(con), add = TRUE)
    if (a > 0) seek(con, a, rw = "read")
    body <- readBin(con, "raw", n = b - a + 1)
    if (length(body) != b - a + 1) .s1_stop(sprintf("%s: short read", p), "singlet_s1_integrity")
    .s1_count(length(body))
    list(status = if (is.null(range)) 200L else 206L, body = body, etag = etag, total = size)
}

# One GET with retries. `range` = c(first, last) byte, inclusive; NULL fetches the whole object.
.s1_get <- function(tr, key, range = NULL, expect_total = NA_real_, if_none_match = NULL, retries = 3L) {
    if (tr$kind == "dir") {
        return(.s1_get_dir(tr, key, range, expect_total, if_none_match))
    }
    .s1_require_curl()
    url <- paste0(tr$base, "/", key)
    last <- NULL
    for (attempt in 0:retries) {
        res <- tryCatch(curl::curl_fetch_memory(url, .s1_handle(range, if_none_match)), error = function(e) e)
        if (inherits(res, "error")) {
            last <- conditionMessage(res)
        } else if (res$status_code %in% c(429L, 500L, 502L, 503L, 504L)) {
            last <- sprintf("HTTP %d", res$status_code)
        } else {
            return(.s1_check_response(res, url, range, expect_total))
        }
        if (attempt < retries) Sys.sleep(min(8, 0.5 * 2^attempt) * (0.75 + 0.5 * (as.numeric(Sys.time()) %% 1)))  # backoff with jitter
    }
    .s1_stop(sprintf("GET %s failed after %d attempts: %s", url, retries + 1L, last))
}

# Several GETs, concurrently when the transport allows. `reqs` is a list of list(key, range, expect_total).
.s1_get_many <- function(tr, reqs, concurrency = 4L) {
    n <- length(reqs)
    one <- function(r) .s1_get(tr, r$key, r$range, if (is.null(r$expect_total)) NA_real_ else r$expect_total)
    if (n == 0L) return(list())
    if (n == 1L || tr$kind != "http" || concurrency <= 1L) {
        return(lapply(reqs, one))
    }
    .s1_require_curl()
    pool <- curl::new_pool(total_con = as.integer(concurrency), host_con = as.integer(concurrency))
    results <- vector("list", n)
    for (i in seq_len(n)) {
        local({
            j <- i
            h <- .s1_handle(reqs[[j]]$range)
            curl::handle_setopt(h, url = paste0(tr$base, "/", reqs[[j]]$key))
            curl::multi_add(h, done = function(res) results[[j]] <<- res, fail = function(msg) NULL, pool = pool)
        })
    }
    curl::multi_run(pool = pool)
    lapply(seq_len(n), function(j) {
        res <- results[[j]]
        r <- reqs[[j]]
        if (!is.null(res) && !(res$status_code %in% c(429L, 500L, 502L, 503L, 504L))) {
            .s1_check_response(res, paste0(tr$base, "/", r$key), r$range, if (is.null(r$expect_total)) NA_real_ else r$expect_total)
        } else {
            one(r)  # a failed transfer is retried on its own, with backoff
        }
    })
}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
.s1_json <- function(raw_or_chr) {
    txt <- if (is.raw(raw_or_chr)) rawToChar(raw_or_chr) else raw_or_chr
    jsonlite::fromJSON(txt, simplifyVector = FALSE)
}

# A study-level legacy file (study_meta.json, ...) from the manifest entry: inline JSON, or stored zstd.
.s1_study_file <- function(study, name) {
    ent <- study$doc$study_files[[name]]
    if (is.null(ent)) return(NULL)
    if (!is.null(ent$json)) return(ent$json)
    if (!is.null(ent$zstd_b64)) {
        return(.s1_json(pz2_zstd_(jsonlite::base64_dec(ent$zstd_b64))))
    }
    NULL
}

.s1_study_object <- function(doc, tr, root, source) {
    samples <- doc$samples
    gsm <- vapply(samples, function(s) s$gsm, character(1))
    names(samples) <- gsm
    structure(list(gse = doc$gse, doc = doc, rev = doc$rev, reference_build = doc$reference_build, packs = doc$packs,
                   samples = samples, gsm_ids = unname(gsm), transport = tr, root = root, source = source),
              class = "singlet_study")
}

#' Open a study's .s1 manifest
#'
#' Fetches \code{s1/m/<GSE>.json} (conditional on the cached ETag, so a repeat costs one tiny request) and returns a
#' study object. Nothing else is downloaded: \code{\link{load}} and \code{\link{singlet_cells}} fetch the sample
#' members they need with HTTP Range requests.
#'
#' @param accession A GEO Series accession (\code{"GSE..."}).
#' @param cache_dir Directory for the manifest and pack cache (default
#'   \code{tools::R_user_dir("singlet", "cache")}, or \code{SINGLET_CACHE_DIR}).
#' @param base Data host (default \code{SINGLET_DATA_BASE} or \code{https://data.singlet.bio}; a directory with the
#'   server layout also works).
#' @param force Re-check the manifest even if it was read less than a minute ago.
#' @return A \code{singlet_study} object (a list with \code{gse}, \code{samples}, \code{gsm_ids}, \code{packs}, ...).
#' @examples
#' \donttest{
#' if (curl::has_internet()) {
#'   st <- singlet_study("GSE46817", cache_dir = file.path(tempdir(), "singlet-example"))
#'   st$gsm_ids
#' }
#' }
#' @seealso \code{\link{load}}, \code{\link{singlet_cells}}
#' @export
singlet_study <- function(accession, cache_dir = NULL, base = NULL, force = FALSE) {
    gse <- toupper(trimws(as.character(accession)[[1L]]))
    if (!grepl("^GSE[0-9]+$", gse)) {
        stop(sprintf("not a GEO Series accession (GSE...): %s", accession), call. = FALSE)
    }
    tr <- .s1_transport(base)
    memo_key <- paste(tr$base, gse, sep = "|")
    hit <- .s1_mem$studies[[memo_key]]
    if (!force && !is.null(hit) && as.numeric(Sys.time()) - hit$time < 60) {
        return(hit$study)
    }
    # DUAL: try s1/m/ first, then the old v2/m/ prefix
    keys <- c(sprintf("s1/m/%s.json", gse), sprintf("v2/m/%s.json", gse))
    key <- keys[1L]
    get_manifest <- function(...) {
        tryCatch(.s1_get(tr, keys[1L], ...), singlet_s1_not_found = function(e) {
            key <<- keys[2L]
            .s1_get(tr, keys[2L], ...)
        })
    }
    root <- .s1_root(cache_dir)
    mdir <- file.path(root, "m")
    mpath <- file.path(mdir, paste0(gse, ".json"))
    epath <- file.path(mdir, paste0(gse, ".etag"))
    not_found <- function(e) {
        .s1_stop(.s1_no_study_message(sprintf(
            "no .s1 manifest for %s at %s",
            gse, paste0(tr$base, "/", keys[1L]))), "singlet_s1_not_found")
    }
    if (tr$kind == "dir") {
        r <- tryCatch(get_manifest(), singlet_s1_not_found = not_found)
        doc <- .s1_json(r$body)
        source <- list(origin = "local", path = file.path(tr$root, key))
    } else {
        etag <- if (file.exists(mpath) && file.exists(epath)) trimws(readLines(epath, warn = FALSE)[1L]) else NULL
        r <- tryCatch(get_manifest(if_none_match = etag),
                      singlet_s1_not_found = not_found,
                      singlet_s1_error = function(e) {
                          if (file.exists(mpath)) {
                              warning(sprintf("could not revalidate %s (%s); using the cached manifest", gse, conditionMessage(e)), call. = FALSE)
                              list(status = -1L)
                          } else stop(e)
                      })
        if (r$status == 304L || r$status == -1L) {
            doc <- .s1_json(paste(readLines(mpath, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))
            source <- list(origin = if (r$status == 304L) "cache-304" else "cache-offline", path = mpath)
        } else {
            doc <- .s1_json(r$body)
            if (!dir.exists(mdir)) dir.create(mdir, recursive = TRUE, showWarnings = FALSE)
            tmp <- paste0(mpath, ".", Sys.getpid(), ".part")
            writeBin(r$body, tmp)
            file.rename(tmp, mpath)
            if (!is.null(r$etag)) writeLines(r$etag, epath) else unlink(epath)
            source <- list(origin = "network", path = mpath)
        }
    }
    if (!(doc$format %in% c("singlet-s1", "singlet-store", "singlet-v2")) || !is.list(doc$samples)) {
        .s1_stop(sprintf("%s: not a singlet-s1 manifest", gse))
    }
    if (isTRUE(doc$manifest_version > 1L)) {
        .s1_stop(sprintf("%s: manifest_version %s is newer than this client supports; update the singlet package", gse, doc$manifest_version))
    }
    study <- .s1_study_object(doc, tr, root, source)
    study$manifest_key <- key
    .s1_mem$studies[[memo_key]] <- list(time = as.numeric(Sys.time()), study = study)
    study
}

#' @export
print.singlet_study <- function(x, ...) {
    cat(sprintf("<singlet_study %s: %d samples, rev %s, %s>\n", x$gse, length(x$gsm_ids), substr(x$rev, 1L, 8L), x$transport$base))
    invisible(x)
}


# ---------------------------------------------------------------------------
# Packs (vocabularies and whitelists): fetched once to the cache, verified by sha256
# ---------------------------------------------------------------------------
.s1_pack_dir <- function(study) {
    d <- file.path(study$root, "ref")
    if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
    d
}

# Raw bytes of the pack whose id starts with `id` (hex, at least 16 characters).
.s1_pack <- function(study, id) {
    id <- tolower(id)
    key <- paste(study$transport$base, id, sep = "|")
    hit <- .s1_mem$packs[[key]]
    if (!is.null(hit)) return(hit)
    dir <- .s1_pack_dir(study)
    files <- list.files(dir, pattern = "^[0-9a-f]{32}\\.pack$")
    cand <- files[startsWith(files, id)]
    raw <- NULL
    if (length(cand) == 1L) {
        raw <- readBin(file.path(dir, cand), "raw", n = file.size(file.path(dir, cand)))
        full <- sub("\\.pack$", "", cand)
        if (!startsWith(pz2_sha256_(raw), full)) {
            unlink(file.path(dir, cand))  # damaged cache entry: fetch it again
            raw <- NULL
        }
    }
    if (is.null(raw)) {
        idx <- .s1_mem$pack_index[[study$transport$base]]
        if (is.null(idx)) {
            idx <- .s1_json(.s1_get_ref(study$transport, "index.json")$body)
            .s1_mem$pack_index[[study$transport$base]] <- idx
        }
        ent <- Filter(function(e) startsWith(e$id, id), idx$packs)
        if (length(ent) != 1L) .s1_stop(sprintf("pack %s is not in the pack index of %s", id, study$transport$base))
        ent <- ent[[1L]]
        raw <- .s1_get_ref(study$transport, sprintf("%s.pack", ent$id))$body
        sha <- pz2_sha256_(raw)
        if (!identical(sha, ent$sha256) || !startsWith(sha, ent$id)) {
            .s1_stop(sprintf("downloaded pack %s fails its checksum", ent$id))
        }
        tmp <- file.path(dir, sprintf("%s.%d.part", ent$id, Sys.getpid()))
        writeBin(raw, tmp)
        file.rename(tmp, file.path(dir, paste0(ent$id, ".pack")))
    }
    .s1_mem$packs[[key]] <- raw
    raw
}

# Vocabulary of a pack: gene table plus the gene (0-based) of every row of the member it describes.
.s1_vocab <- function(study, id) {
    key <- paste(study$transport$base, tolower(id), sep = "|")
    hit <- .s1_mem$vocab[[key]]
    if (!is.null(hit)) return(hit)
    raw <- .s1_pack(study, id)
    info <- pz2_pack_info_(raw)
    hdr <- .s1_json(info$header)
    if (info$kind == 3L) {  # composite (barnyard): the parts' vocabularies side by side
        parts <- lapply(info$parts, function(p) .s1_vocab(study, p$id))
        offs <- c(0L, cumsum(vapply(parts, function(p) length(p$gene_id), integer(1))))
        v <- list(gene_id = unlist(lapply(parts, function(p) p$gene_id)), gene_name = unlist(lapply(parts, function(p) p$gene_name)),
                  row_gene = unlist(lapply(seq_along(parts), function(k) parts[[k]]$row_gene + offs[[k]])), member = hdr$member)
    } else if (nzchar(info$derived_from)) {  # intron pack: derived from the exon pack of the same build
        ex <- .s1_vocab(study, info$derived_from)
        v <- list(gene_id = ex$gene_id, gene_name = ex$gene_name, nex = ex$nex, member = "intron",
                  row_gene = rep.int(seq_along(ex$nex) - 1L, ex$nex - 1L))
    } else {
        g <- pz2_pack_genes_(raw)
        v <- list(gene_id = g$gene_id, gene_name = g$gene_name, nex = g$nex, member = "exon",
                  row_gene = rep.int(seq_along(g$nex) - 1L, g$nex))
    }
    .s1_mem$vocab[[key]] <- v
    v
}


# ---------------------------------------------------------------------------
# Cells tables
# ---------------------------------------------------------------------------
.s1_cells <- function(study, raw, columns) {
    cells <- pz2_cells_(raw, as.character(columns), NULL)
    bc <- cells$columns[["barcode"]]
    if (is.list(bc)) {  # coded against a whitelist pack
        wl <- .s1_pack(study, bc$whitelist)
        cells$columns[["barcode"]] <- pz2_whitelist_barcodes_(wl, bc$index)
    }
    cells
}


# ---------------------------------------------------------------------------
# Modalities (the same vocabulary as python/singlet/s1/client.py MODALITIES)
# ---------------------------------------------------------------------------
.s1_modality_members <- list(
    counts = c("exon_counts.1pz", "intron_counts.1pz"),  # counts assay = spliced + unspliced
    spliced = "exon_counts.1pz",
    unspliced = "intron_counts.1pz",
    splice_junctions = "sj_counts.1pz",                    # altExp: reads per junction
    psi = "sj_counts.1pz",                                 # altExp: percent spliced in, computed from the junctions
    mt = "mt_heteroplasmy.1pz",
    vdj = "vdj_gene_usage.1pz")
.s1_modality_alias <- c(x = "counts", exon = "spliced", intron = "unspliced", junctions = "splice_junctions",
                        sj = "splice_junctions", splice_psi = "psi", mt_heteroplasmy = "mt", vdj_gene_usage = "vdj")
.s1_gene_members <- c("exon_counts.1pz", "intron_counts.1pz")
.s1_gene_modalities <- c("counts", "spliced", "unspliced")

# Canonical names (vocabulary order) of `modalities`; NULL keeps counts + spliced + unspliced (counts alone without layers).
.s1_normalize_modalities <- function(modalities, layers = TRUE) {
    if (is.null(modalities)) return(if (layers) .s1_gene_modalities else "counts")
    k <- tolower(trimws(as.character(modalities)))
    k <- ifelse(k %in% names(.s1_modality_alias), unname(.s1_modality_alias[k]), k)
    bad <- setdiff(k, names(.s1_modality_members))
    if (length(bad) > 0L) {
        .s1_stop(sprintf("unknown modality %s; the vocabulary is %s", paste(sQuote(bad, FALSE), collapse = ", "),
                         paste(names(.s1_modality_members), collapse = ", ")), "singlet_s1_modality_unavailable")
    }
    if (length(k) == 0L) .s1_stop("modalities is empty", "singlet_s1_modality_unavailable")
    intersect(names(.s1_modality_members), k)
}

# A readable pz2 member with at least one row and column (stubs and empty legacy members are absent).
.s1_member_has_data <- function(sm, name) {
    m <- sm$members[[name]]
    !is.null(m) && identical(as.integer(m$kind), 1L) && m$len > 0 && !identical(m$m, 0L) && !identical(m$n, 0L)
}

# The modalities one sample carries (counts needs only the exon matrix: older samples have no intron matrix).
.s1_sample_modalities <- function(sm) {
    Filter(function(k) {
        mem <- if (k == "counts") "exon_counts.1pz" else .s1_modality_members[[k]]
        all(vapply(mem, function(n) .s1_member_has_data(sm, n), NA))
    }, names(.s1_modality_members))
}

.s1_check_modalities <- function(study, gsms, mods) {
    have <- unique(unlist(lapply(gsms, function(g) .s1_sample_modalities(study$samples[[g]]))))
    have <- intersect(names(.s1_modality_members), have)
    missing <- setdiff(mods, have)
    if (length(missing) > 0L) {
        .s1_stop(sprintf("%s: no requested sample has %s; %s %s", study$gse, paste(missing, collapse = ", "),
                         if (length(gsms) > 1L) "these samples have" else paste(gsms, "has"), paste(have, collapse = ", ")),
                 "singlet_s1_modality_unavailable")
    }
    invisible(TRUE)
}

.s1_members_for <- function(mods) unique(unlist(.s1_modality_members[mods], use.names = FALSE))

#' Modalities of a per-sample study
#'
#' The names \code{load(modalities = )} and \code{singlet_cells(modalities = )} accept that a study's samples carry,
#' read from the manifest (no sample bytes are fetched).
#'
#' @param x A \code{singlet_study} or a GEO Series accession.
#' @param gsm Optional sample accession: only that sample's modalities.
#' @return A character vector: a subset of \code{counts}, \code{spliced}, \code{unspliced}, \code{splice_junctions},
#'   \code{psi}, \code{mt}, \code{vdj}.
#' @export
singlet_study_modalities <- function(x, gsm = NULL) {
    study <- if (inherits(x, "singlet_study")) x else singlet_study(x)
    gsms <- if (is.null(gsm)) study$gsm_ids else toupper(gsm)
    have <- unique(unlist(lapply(gsms, function(g) .s1_sample_modalities(study$samples[[g]]))))
    intersect(names(.s1_modality_members), have)
}

# Junction events (python/singlet/s1/recipes_psi.py build_events / junction_events): junctions "chrom:start-end" that share a
# donor (chrom, start) or an acceptor (chrom, end) form an event when there are at least two; each junction belongs to the first
# of its events in bytewise order of the event id. Returns the 1-based event of every junction (NA: no event).
.s1_junction_events <- function(names) {
    n <- length(names)
    c <- regexpr(":", names, fixed = TRUE)
    rest <- substring(names, c + 1L)
    d <- regexpr("-", rest, fixed = TRUE)
    num <- function(v) {
        pat <- "^[[:space:]]*[+]?([0-9]+).*$"
        out <- rep(NA_real_, length(v))
        hit <- grepl(pat, v)
        out[hit] <- as.numeric(sub(pat, "\\1", v[hit])) %% 2^32
        out
    }
    st <- num(substr(rest, 1L, d - 1L))
    en <- num(substring(rest, d + 1L))
    ok <- c > 0L & d > 0L & !is.na(st) & !is.na(en)
    chrom <- substr(names, 1L, c - 1L)
    ids <- character(0)
    jid <- list()
    for (kind in c("D", "A")) {
        pos <- if (kind == "D") st else en
        key <- ifelse(ok, sprintf("%s:%.0f-%s", chrom, pos, kind), NA_character_)
        tab <- table(key[ok])
        ev <- names(tab)[tab >= 2L]
        jid[[kind]] <- ifelse(key %in% ev, key, NA_character_)
        ids <- c(ids, ev)
    }
    ids <- sort(unique(ids), method = "radix")
    rD <- match(jid$D, ids)
    rA <- match(jid$A, ids)
    out <- pmin(rD, rA, na.rm = TRUE)
    out[is.na(rD) & is.na(rA)] <- NA_integer_
    as.integer(out)
}

# True PSI of a junctions x cells dgCMatrix: each count over the sum of its event's counts in that cell (cells whose event sum
# is 0 and junctions in no event have no entry).
.s1_psi <- function(m, names) {
    je <- .s1_junction_events(names)
    i <- m@i + 1L
    j <- rep.int(seq_len(ncol(m)), diff(m@p))
    ev <- je[i]
    keep <- !is.na(ev)
    i <- i[keep]; j <- j[keep]; x <- m@x[keep]; ev <- ev[keep]
    grp <- match(j * (length(names) + 1) + ev, unique(j * (length(names) + 1) + ev))
    esum <- rowsum(x, grp, reorder = FALSE)[grp]
    nz <- esum > 0
    Matrix::sparseMatrix(i = i[nz], j = j[nz], x = x[nz] / esum[nz], dims = dim(m), dimnames = dimnames(m))
}


# ---------------------------------------------------------------------------
# Reading samples
# ---------------------------------------------------------------------------
.s1_coalesce <- function(off, len, gap) {
    ord <- order(off)
    groups <- list()
    for (i in ord) {
        if (len[[i]] <= 0) next
        n <- length(groups)
        if (n > 0L && off[[i]] - groups[[n]]$end <= gap) {
            groups[[n]]$end <- max(groups[[n]]$end, off[[i]] + len[[i]])
            groups[[n]]$idx <- c(groups[[n]]$idx, i)
        } else {
            groups[[n + 1L]] <- list(start = off[[i]], end = off[[i]] + len[[i]], idx = i)
        }
    }
    groups
}

# Object reader for one sample: a function(off, len) -> list of raw vectors, coalescing near ranges.
.s1_reader <- function(study, sm, concurrency = 4L) {
    local_path <- file.path(study$root, "s", sm$gsm, basename(sm$key))
    if (study$transport$kind == "dir") local_path <- file.path(study$transport$root, sm$key)
    use_local <- file.exists(local_path) && file.size(local_path) == sm$bytes
    function(off, len, gap = 2 * 1024^2) {
        groups <- .s1_coalesce(off, len, gap)
        bufs <- if (use_local) {
            lapply(groups, function(g) .s1_get_dir(list(kind = "dir", root = dirname(local_path)), basename(local_path), c(g$start, g$end - 1), sm$bytes, NULL)$body)
        } else {
            reqs <- lapply(groups, function(g) list(key = sm$key, range = c(g$start, g$end - 1), expect_total = as.numeric(sm$bytes)))
            lapply(.s1_get_many(study$transport, reqs, concurrency), function(r) r$body)
        }
        out <- vector("list", length(off))
        for (k in seq_along(groups)) {
            g <- groups[[k]]
            for (i in g$idx) {
                a <- off[[i]] - g$start + 1
                out[[i]] <- if (a == 1 && len[[i]] == length(bufs[[k]])) bufs[[k]] else bufs[[k]][seq.int(a, length.out = len[[i]])]
            }
        }
        out
    }
}

# Column runs [c0, c1) aligned to chunk boundaries that cover `sel` (0-based, ascending); touching chunks share a run.
.s1_runs <- function(sel, first_col) {
    if (length(sel) == 0L) return(list())
    ch <- findInterval(sel, first_col, left.open = FALSE)  # 1-based chunk index
    u <- sort(unique(ch))
    runs <- list()
    for (c in u) {
        n <- length(runs)
        if (n > 0L && c <= runs[[n]][[2L]] + 1L) runs[[n]][[2L]] <- c else runs[[n + 1L]] <- c(c, c)
    }
    lapply(runs, function(r) c(first_col[[r[[1L]]]], first_col[[r[[2L]] + 1L]]))
}

.s1_dgc <- function(res) {
    loadNamespace("Matrix")  # the dgCMatrix class is defined there
    methods::new("dgCMatrix", i = res$i, p = res$p, x = res$x, Dim = c(res$nrow, res$ncol), Dimnames = list(NULL, NULL))
}

.s1_empty_gene_matrix <- function(n_genes, n_cells) {
    loadNamespace("Matrix")
    methods::new("dgCMatrix", i = integer(0), p = integer(n_cells + 1L), x = numeric(0), Dim = c(as.integer(n_genes), as.integer(n_cells)),
                 Dimnames = list(NULL, NULL))
}

# stored-row -> final gene map (0-based, -1 drops) of a matrix member, from its header.
.s1_member_gene_map <- function(study, hdr, member_raw, axis, keep_map) {
    if (!identical(hdr$row_names_kind, "pack")) {
        nm <- pz2_names_(member_raw, TRUE)
        gm <- match(sub("_.*$", "", nm$names), axis$gene_id) - 1L
        gm[is.na(gm)] <- -1L
    } else {
        v <- .s1_vocab(study, hdr$row_names_id)
        if (length(v$row_gene) != hdr$m) {
            .s1_stop(sprintf("vocabulary pack %s has %d rows, the matrix has %.0f", hdr$row_names_id, length(v$row_gene), hdr$m))
        }
        gm <- v$row_gene
        if (!identical(v$gene_id, axis$gene_id)) {  # another vocabulary: translate by gene id
            to_axis <- match(v$gene_id, axis$gene_id) - 1L
            to_axis[is.na(to_axis)] <- -1L
            gm <- ifelse(gm >= 0L, to_axis[gm + 1L], -1L)
        }
    }
    if (!is.null(keep_map)) gm <- ifelse(gm >= 0L, keep_map[gm + 1L], -1L)
    as.integer(gm)
}

# One sample: cells table, then the wanted matrices (`members`; default the exon matrix and, when present, the intron matrix)
# for the selected cells: exon and intron decoded to genes, the others (junctions, V(D)J, ...) as feature x cell matrices with
# their row names. Only those members' bytes are read. `sel_fn(cells)` returns the selected matrix columns (0-based,
# ascending). Returns a list, or list(skipped = reason).
.s1_read_sample <- function(study, gsm, axis, keep_map, n_genes, sel_fn, want_cols, threads = 0L, whole_max = 48 * 1024^2,
                            concurrency = 4L, verify_digest = TRUE, members = NULL) {
    sm <- study$samples[[gsm]]
    mem <- sm$members
    cm <- mem[["cells"]]
    em <- mem[["exon_counts.1pz"]]
    if (is.null(members)) members <- .s1_gene_members
    if (is.null(cm)) .s1_stop(sprintf("%s: the sample object has no cells table", gsm))
    if ("exon_counts.1pz" %in% members && (is.null(em) || em$len == 0)) return(list(skipped = "exon_counts.1pz is missing"))
    present <- Filter(function(n) n == "exon_counts.1pz" || .s1_member_has_data(sm, n), members)
    if (length(present) == 0L) {
        return(list(skipped = if (length(members) == 1L) paste(members, "is missing") else
            sprintf("none of %s is present", paste(members, collapse = ", "))))
    }
    reader <- .s1_reader(study, sm, concurrency)
    total <- cm$len + sum(vapply(present, function(n) as.numeric(mem[[n]]$len), numeric(1)))
    local_obj <- file.exists(file.path(study$root, "s", sm$gsm, basename(sm$key))) || study$transport$kind == "dir"
    whole <- local_obj || total <= whole_max
    names_m <- c("cells", present)
    if (whole) {
        # whole members are merged only across alignment padding: another member's body never rides along
        bl <- reader(vapply(names_m, function(n) as.numeric(mem[[n]]$off), numeric(1)), vapply(names_m, function(n) as.numeric(mem[[n]]$len), numeric(1)),
                     gap = 63)
        names(bl) <- names_m
        cells_raw <- bl[["cells"]]
    } else {
        pre_len <- vapply(names_m, function(n) if (n == "cells") as.numeric(mem[[n]]$len) else min(as.numeric(mem[[n]]$len), 512 * 1024), numeric(1))
        bl <- reader(vapply(names_m, function(n) as.numeric(mem[[n]]$off), numeric(1)), pre_len, gap = 256 * 1024)
        names(bl) <- names_m
        cells_raw <- bl[["cells"]]
    }
    cells <- .s1_cells(study, cells_raw, want_cols)
    n_rows <- cells$n_rows
    if (n_rows == 0) return(list(skipped = "count matrix has 0 cells"))
    sel <- as.integer(sel_fn(cells))
    if (length(sel) == 0L) return(list(skipped = "0 cells called", n_rows = n_rows))
    if (is.unsorted(sel, strictly = TRUE) || sel[[1L]] < 0L || sel[[length(sel)]] >= n_rows) {
        .s1_stop(sprintf("%s: column selection must be ascending and inside the matrix", gsm))
    }
    cols <- lapply(cells$columns, function(v) if (is.list(v)) v else v[sel + 1L])
    c_hi <- sel[[length(sel)]] + 1L
    parts <- list(exon = NULL, intron = NULL)
    features <- list()
    for (name in present) {
        gene_level <- name %in% .s1_gene_members
        mref <- mem[[name]]
        if (is.null(mref) || mref$len == 0) next
        buf <- bl[[name]]
        hdr <- pz2_header_(buf)
        if (hdr$n != n_rows) .s1_stop(sprintf("%s: %s has %.0f columns, the cells table has %.0f", gsm, name, hdr$n, n_rows))
        if (hdr$m == 0) next
        if (verify_digest && !is.null(mref$digest) && !identical(hdr$digest, tolower(mref$digest))) {
            .s1_stop(sprintf("%s: %s header digest %s != manifest %s", gsm, name, hdr$digest, mref$digest))
        }
        if (!whole && length(buf) < hdr$prefix_len) {  # the first read was shorter than the prefix
            buf <- reader(mref$off, min(mref$len, hdr$prefix_len), gap = 0)[[1L]]
        }
        if (gene_level) {
            mode <- "gene"
            gm <- .s1_member_gene_map(study, hdr, buf, axis, keep_map)
            ng <- n_genes
        } else {
            if (!identical(hdr$row_names_kind, "inline")) {
                .s1_stop(sprintf("%s: %s names its rows from a pack (%s), which the R reader does not resolve yet", gsm, name,
                                 hdr$row_names_kind))
            }
            mode <- "canonical"
            gm <- NULL
            ng <- 0L
        }
        if (whole) {
            res <- pz2_decode_(buf, numeric(0), list(), 0, c_hi, mode, sel, gm, ng, threads, "auto")
            part <- .s1_dgc(res)
        } else {
            first_col <- pz2_chunk_first_cols_(buf)
            runs <- .s1_runs(sel, first_col)
            plans <- lapply(runs, function(r) pz2_plan_(buf, r[[1L]], r[[2L]]))
            got <- reader(vapply(plans, function(p) mref$off + p$byte_off, numeric(1)), vapply(plans, function(p) p$byte_len, numeric(1)), gap = 256 * 1024)
            pieces <- lapply(seq_along(runs), function(k) {
                r <- runs[[k]]
                s_run <- sel[sel >= r[[1L]] & sel < r[[2L]]]
                .s1_dgc(pz2_decode_(buf, plans[[k]]$byte_off, list(got[[k]]), r[[1L]], r[[2L]], mode, s_run - r[[1L]], gm, ng, threads, "auto"))
            })
            part <- if (length(pieces) == 1L) pieces[[1L]] else do.call(cbind, pieces)
        }
        if (gene_level) {
            parts[[if (name == "exon_counts.1pz") "exon" else "intron"]] <- part
        } else {
            rownames(part) <- pz2_names_(buf, TRUE)$names
            features[[name]] <- part
        }
    }
    list(gsm = gsm, sel = sel, cols = cols, exon = parts$exon, intron = parts$intron, features = features, n_rows = n_rows)
}

.s1_sample_empty <- function(r) is.null(r$exon) && is.null(r$intron) && length(r$features) == 0L

# Assays and altExps of the selected modalities from per-sample reads `ok` (each with `ord`, the cell order, and `cn`, the cell
# names). counts / spliced / unspliced are gene assays; splice_junctions, psi, mt and vdj are altExps (SummarizedExperiment) on
# the union of the samples' feature names, with empty columns for samples that lack the member.
.s1_assemble <- function(ok, mods, axis, n_genes, explicit) {
    loadNamespace("Matrix")
    ord_cols <- function(m, r) if (identical(r$ord, seq_along(r$ord))) m else m[, r$ord, drop = FALSE]
    cn <- unlist(lapply(ok, `[[`, "cn"), use.names = FALSE)
    assays <- list()
    gene <- function(r, what) {
        m <- r[[what]]
        if (is.null(m)) NULL else ord_cols(m, r)
    }
    if ("counts" %in% mods) {
        assays$counts <- do.call(cbind, unname(lapply(ok, function(r) {
            sp <- gene(r, "exon"); un <- gene(r, "intron")
            if (is.null(sp) && is.null(un)) .s1_empty_gene_matrix(n_genes, length(r$sel))
            else if (is.null(un)) sp else if (is.null(sp)) un else sp + un
        })))
    }
    for (a in intersect(c("spliced", "unspliced"), mods)) {
        what <- if (a == "spliced") "exon" else "intron"
        lacking <- vapply(ok, function(r) is.null(r[[what]]), NA)
        if (any(lacking) && explicit) {
            warning(sprintf("assay %s omitted: %s have no %s", a, paste(vapply(ok[lacking], `[[`, "", "gsm"), collapse = ", "),
                            .s1_modality_members[[a]]), call. = FALSE)
            next
        }
        # the default request keeps the assay, with empty columns for samples that lack the matrix (as before modalities=)
        assays[[a]] <- do.call(cbind, unname(lapply(ok, function(r) {
            m <- gene(r, what)
            if (is.null(m)) .s1_empty_gene_matrix(n_genes, length(r$sel)) else m
        })))
    }
    gene_mods <- intersect(mods, .s1_gene_modalities)
    for (k in names(assays)) {
        a <- methods::as(assays[[k]], "CsparseMatrix")
        dimnames(a) <- list(axis$gene_id, cn)
        assays[[k]] <- a
    }
    alt <- list()
    for (k in setdiff(mods, .s1_gene_modalities)) {
        member <- .s1_modality_members[[k]]
        mats <- lapply(ok, function(r) {
            m <- r$features[[member]]
            if (is.null(m)) return(NULL)
            if (k == "psi") m <- .s1_psi(m, rownames(m))
            ord_cols(m, r)
        })
        lacking <- vapply(mats, is.null, NA)
        if (any(lacking)) {
            warning(sprintf("%s: %s have no %s; their columns of altExp(\"%s\") are empty", k,
                            paste(vapply(ok[lacking], `[[`, "", "gsm"), collapse = ", "), member, k), call. = FALSE)
        }
        feats <- unique(unlist(lapply(mats, rownames), use.names = FALSE))
        blocks <- lapply(seq_along(ok), function(s) {
            m <- mats[[s]]
            n <- length(ok[[s]]$sel)
            if (is.null(m)) return(Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = c(length(feats), n)))
            t <- methods::as(m, "TsparseMatrix")
            Matrix::sparseMatrix(i = match(rownames(m), feats)[t@i + 1L], j = t@j + 1L, x = t@x, dims = c(length(feats), n))
        })
        X <- methods::as(do.call(cbind, blocks), "CsparseMatrix")
        dimnames(X) <- list(feats, cn)
        alt[[k]] <- SummarizedExperiment::SummarizedExperiment(assays = stats::setNames(list(X), k))
    }
    rowdata <- S4Vectors::DataFrame(gene_name = if (length(gene_mods)) axis$gene_name else character(0))
    rownames(rowdata) <- if (length(gene_mods)) axis$gene_id else character(0)
    list(assays = assays, alt = alt, rowdata = rowdata, cn = cn, gene_mods = gene_mods)
}

.s1_make_sce <- function(parts, coldata) {
    assays <- parts$assays
    if (length(assays) == 0L) {  # no gene modality: an empty gene axis, the data are in altExps
        assays <- list()
        sce <- SingleCellExperiment::SingleCellExperiment(assays = assays, colData = S4Vectors::DataFrame(coldata, check.names = FALSE))
    } else {
        sce <- SingleCellExperiment::SingleCellExperiment(assays = assays, colData = S4Vectors::DataFrame(coldata, check.names = FALSE),
                                                          rowData = parts$rowdata)
    }
    for (k in names(parts$alt)) SingleCellExperiment::altExp(sce, k) <- parts$alt[[k]]
    sce
}

# Order of the selected cells: the legacy reader lists called cells as cell_calls.tsv does (STARsolo: sorted barcodes).
.s1_cell_order <- function(cols, order = c("legacy", "barcode", "column")) {
    order <- match.arg(order)
    n <- length(cols$barcode)
    if (order == "column" || n < 2L) return(seq_len(n))
    if (order == "legacy") {
        if (!is.null(cols$cc_pos)) return(base::order(cols$cc_pos))
        if (!is.null(cols$cc_listed) || !is.null(cols$cc_pvalue)) return(seq_len(n))  # EmptyDrops lists ascending columns
    }
    base::order(cols$barcode, method = "radix")
}


# ---------------------------------------------------------------------------
# Study -> SingleCellExperiment
# ---------------------------------------------------------------------------
.s1_axis <- function(study) {
    v <- .s1_vocab(study, study$packs$exon)
    list(gene_id = v$gene_id, gene_name = v$gene_name)
}

.s1_read_study <- function(study, gsms = NULL, cell_order = "legacy", threads = 0L, concurrency = 4L, whole_max = 48 * 1024^2,
                           modalities = NULL, cell_type_level = 2L, cell_type_min_score = 0, include_unvalidated = FALSE) {
    .require_sce("load()")
    mods <- .s1_normalize_modalities(modalities)
    gene_axis <- any(mods %in% .s1_gene_modalities)
    axis <- if (gene_axis) .s1_axis(study) else list(gene_id = character(0), gene_name = character(0))
    n_genes <- length(axis$gene_id)
    all_gsms <- study$gsm_ids
    skipped <- character(0)
    if (is.null(gsms)) {
        targets <- all_gsms
    } else {
        gsms <- unique(as.character(gsms))
        absent <- setdiff(gsms, all_gsms)
        skipped[absent] <- "not in this study; excluded when it was packed"
        targets <- gsms[gsms %in% all_gsms]
    }
    n_asked <- length(targets) + length(skipped)
    if (!is.null(modalities) && length(targets) > 0L) .s1_check_modalities(study, targets, mods)
    members <- .s1_members_for(mods)
    study_meta <- .s1_study_file(study, "study_meta.json")
    gsm_meta_map <- if (!is.null(study_meta)) study_meta$gsm_meta else NULL

    res <- lapply(targets, function(g) {
        tryCatch(
            .s1_read_sample(study, g, axis, NULL, n_genes, function(cells) which(cells$columns$called) - 1L,
                            character(0), threads, whole_max, concurrency, members = members),
            singlet_s1_integrity = function(e) stop(e),
            error = function(e) list(skipped = paste("unreadable:", conditionMessage(e)))
        )
    })
    names(res) <- targets
    ok <- list(); coldata_rows <- list()
    for (g in targets) {
        r <- res[[g]]
        if (is.null(r$skipped) && .s1_sample_empty(r)) r$skipped <- "count matrix has no usable features"
        if (!is.null(r$skipped)) {
            skipped[[g]] <- r$skipped
            next
        }
        r$ord <- .s1_cell_order(r$cols, cell_order)
        r$cn <- paste0(g, "_", r$cols$barcode[r$ord])
        ok[[g]] <- r
        info <- if (!is.null(gsm_meta_map) && g %in% names(gsm_meta_map)) gsm_meta_map[[g]] else list()
        n <- length(r$cn)
        cols <- lapply(r$cols, function(v) if (is.list(v)) v else v[r$ord])
        lab <- .s1_label_columns(cols, n, .s1_label_categories(study$samples[[g]]$members), cell_type_level,
                                 cell_type_min_score, include_unvalidated)
        geo <- .bundle_obs_frame(g, info, n)
        coldata_rows[[g]] <- c(list(gsm_id = rep(g, n)), lab, as.list(geo[setdiff(names(geo), "gsm_id")]))
    }
    if (length(ok) == 0L) {
        detail <- if (length(skipped) > 0L) paste0(": ", .describe_skipped(skipped)) else ""
        stop(sprintf("study %s contained no loadable cells%s", study$gse, detail), call. = FALSE)
    }
    if (length(skipped) > 0L) {
        warning(sprintf("%s: skipped %d of %d sample(s) with no usable cells: %s", study$gse, length(skipped), n_asked, .describe_skipped(skipped)),
                call. = FALSE)
    }
    parts <- .s1_assemble(ok, mods, axis, n_genes, explicit = !is.null(modalities))
    coldata <- .s1_label_rbind(unname(coldata_rows))
    rownames(coldata) <- parts$cn
    sce <- .s1_make_sce(parts, coldata)
    S4Vectors::metadata(sce)$manifest <- list(
        schema_version = "2.0", gse_id = study$gse, gsm_ids = names(ok), reference_build = study$reference_build, n_gsms = length(ok),
        created_at = study$doc$created_at, format = "singlet-s1", rev = study$rev)
    S4Vectors::metadata(sce)$modalities <- mods
    if (!is.null(study_meta)) S4Vectors::metadata(sce)$study_meta <- study_meta
    S4Vectors::metadata(sce)$singlet_bundle_path <- paste0(study$transport$base, "/", if (is.null(study$manifest_key)) sprintf("s1/m/%s.json", study$gse) else study$manifest_key)
    S4Vectors::metadata(sce)$skipped_samples <- data.frame(
        gsm_id = as.character(names(skipped)), reason = unname(skipped), stringsAsFactors = FALSE)
    sce
}


# ---------------------------------------------------------------------------
# Cell-level slices
# ---------------------------------------------------------------------------

#' Cells of a study that satisfy a condition, without downloading the rest
#'
#' Reads the (small) cell table of each requested sample, evaluates \code{where} on its columns
#' (\code{called}, \code{n_umi_exon}, \code{n_umi_intron}, \code{n_feat_exon}, \code{n_feat_intron}, ..., and the per-cell labels
#' of \code{\link{singlet_labels}}, e.g. \code{cell_type == "T cell" & sex_chrom == "female"}) and fetches only the
#' matrix chunks that contain the selected cells (a small sample is fetched whole, in one request). Counts are at gene level
#' (spliced + unspliced), as in \code{\link{load}}.
#'
#' @param x A GEO Series accession (\code{"GSE..."}) or a sample accession (\code{"GSM..."}, which selects that sample of
#'   its parent Series), or a \code{singlet_study} object.
#' @param where An unevaluated condition on the cell-table columns, e.g. \code{n_umi_exon > 5000 & called}; or a function of
#'   the cell-table data frame returning a logical vector; or \code{NULL} for every cell.
#' @param samples GSM accessions to restrict the Series to (default all).
#' @param columns Columns to keep in \code{colData} (default all; see \code{\link{singlet_labels}}).
#' @param layers Also return \code{spliced} and \code{unspliced} assays.
#' @param modalities Which matrices to fetch (default \code{"counts"}, plus the layers with \code{layers = TRUE}); see
#'   \code{\link{load}}. Only the chunk runs of those members that hold the selected cells are downloaded.
#' @param max_cells Keep at most this many cells per sample (the largest libraries first).
#' @param as \code{"sce"} (default) or \code{"seurat"}.
#' @param cache_dir Cache directory (see \code{\link{load}}).
#' @param cell_type_level,cell_type_min_score Deepest level of \code{cell_type} (default 2) and an extra minimum path score for
#'   it (default 0, none); see \code{\link{singlet_labels}}.
#' @param include_unvalidated Also keep \code{qc_doublet_score} (unvalidated; local outputs only, published objects never carry it).
#' @return A \code{SingleCellExperiment} (or \code{Seurat}) with one column per selected cell named
#'   \code{<GSM>_<barcode>}; \code{colData} has \code{gsm_id}, \code{matrix_column} (1-based column in the sample's
#'   matrix) and the per-cell columns (\code{\link{singlet_labels}}).
#' @examples
#' \donttest{
#' if (curl::has_internet() && requireNamespace("SingleCellExperiment", quietly = TRUE)) {
#'   deep <- singlet_cells("GSE46817", n_umi_exon > 500, samples = "GSM1138786",
#'                         cache_dir = file.path(tempdir(), "singlet-example"))
#' }
#' }
#' @seealso \code{\link{load}}, \code{\link{singlet_study}}
#' @export
singlet_cells <- function(x, where = NULL, samples = NULL, columns = NULL, layers = FALSE, max_cells = NULL,
                          as = c("sce", "seurat"), cache_dir = NULL, modalities = NULL, cell_type_level = 2L,
                          cell_type_min_score = 0, include_unvalidated = FALSE) {
    as <- match.arg(as)
    cond <- substitute(where)
    caller <- parent.frame()
    # `where` is normally a bare expression over the cell-table columns; a function or a string are accepted too.
    where_val <- tryCatch(where, error = function(e) NULL)
    if (inherits(x, "singlet_study")) {
        study <- x
    } else {
        acc <- .as_accession(as.character(x)[[1L]])
        if (is.na(acc)) stop(sprintf("not a GEO accession (GSE... or GSM...): %s", x), call. = FALSE)
        if (startsWith(acc, "GSM")) {
            samples <- if (is.null(samples)) acc else samples
        }
        study <- singlet_study(.gse_of(acc, cache = .singlet_cache_dir(cache_dir)), cache_dir = cache_dir)
    }
    .require_sce("singlet_cells()")
    gsms <- if (is.null(samples)) study$gsm_ids else as.character(samples)
    bad <- setdiff(gsms, study$gsm_ids)
    if (length(bad) > 0L) stop(sprintf("sample(s) %s not in %s", paste(bad, collapse = ", "), study$gse), call. = FALSE)
    mods <- .s1_normalize_modalities(modalities, layers)
    if (!is.null(modalities)) .s1_check_modalities(study, gsms, mods)
    axis <- if (any(mods %in% .s1_gene_modalities)) .s1_axis(study) else list(gene_id = character(0), gene_name = character(0))
    n_genes <- length(axis$gene_id)
    keep_fun <- function(cells, gsm) {
        df <- as.data.frame(.s1_label_columns(cells$columns[setdiff(names(cells$columns), "barcode")], cells$n_rows,
                                              .s1_label_categories(study$samples[[gsm]]$members), cell_type_level,
                                              cell_type_min_score, include_unvalidated = TRUE),
                            stringsAsFactors = FALSE, check.names = FALSE)
        m <- if (is.function(where_val)) {
            where_val(df)
        } else if (is.character(where_val) && length(where_val) == 1L) {
            eval(parse(text = where_val), df, caller)
        } else if (is.null(cond)) {
            rep(TRUE, nrow(df))
        } else {
            eval(cond, df, caller)
        }
        idx <- which(as.logical(m)) - 1L
        if (!is.null(max_cells)) idx <- utils::head(idx, max_cells)
        idx
    }
    res <- lapply(gsms, function(g) {
        .s1_read_sample(study, g, axis, NULL, n_genes, function(cells) keep_fun(cells, g), character(0), 0L,
                        members = .s1_members_for(mods))
    })
    names(res) <- gsms
    ok <- Filter(function(r) is.null(r$skipped) && !.s1_sample_empty(r), res)
    if (length(ok) == 0L) {
        return(SingleCellExperiment::SingleCellExperiment(assays = list(counts = .s1_empty_gene_matrix(n_genes, 0L))))
    }
    ok <- lapply(ok, function(r) {
        r$ord <- seq_along(r$sel)
        r$cn <- paste0(r$gsm, "_", r$cols$barcode)
        r
    })
    parts <- .s1_assemble(ok, mods, axis, n_genes, explicit = !is.null(modalities))
    cn <- parts$cn
    cd <- .s1_label_rbind(unname(lapply(ok, function(r) {
        n <- length(r$sel)
        c(list(gsm_id = rep(r$gsm, n), matrix_column = r$sel + 1L),
          .s1_label_columns(r$cols, n, .s1_label_categories(study$samples[[r$gsm]]$members), cell_type_level,
                            cell_type_min_score, include_unvalidated, columns))
    })))
    rownames(cd) <- cn
    sce <- .s1_make_sce(parts, cd)
    S4Vectors::metadata(sce)$singlet_s1 <- list(gse = study$gse, rev = study$rev, modalities = mods)
    if (as == "seurat") return(.sce_to_seurat(sce))
    sce
}


# ---------------------------------------------------------------------------
# Offline copy of a study
# ---------------------------------------------------------------------------
.s1_download <- function(gse, cache_dir = NULL, verify = TRUE) {
    study <- singlet_study(gse, cache_dir = cache_dir, force = TRUE)
    root <- study$root
    for (g in study$gsm_ids) {
        sm <- study$samples[[g]]
        dest <- file.path(root, "s", g, basename(sm$key))
        if (file.exists(dest) && file.size(dest) == sm$bytes) next
        dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
        tmp <- paste0(dest, ".", Sys.getpid(), ".part")
        con <- file(tmp, "wb")
        pos <- 0
        step <- 32 * 1024^2
        ok <- tryCatch({
            while (pos < sm$bytes) {
                n <- min(step, sm$bytes - pos)
                writeBin(.s1_get(study$transport, sm$key, c(pos, pos + n - 1), expect_total = as.numeric(sm$bytes))$body, con)
                pos <- pos + n
            }
            TRUE
        }, finally = close(con))
        if (verify && !is.null(sm$sha256)) {
            raw <- readBin(tmp, "raw", n = sm$bytes)
            if (!identical(pz2_sha256_(raw), tolower(sm$sha256))) {
                unlink(tmp)
                .s1_stop(sprintf("%s: sha256 of the downloaded object does not match the manifest", g), "singlet_s1_integrity")
            }
        }
        file.rename(tmp, dest)
    }
    for (id in unique(unlist(study$packs))) if (!is.null(id) && nzchar(id)) tryCatch(.s1_pack(study, id), error = function(e) NULL)
    # the pack index too, so a load of the local copy can resolve any pack without the network
    ip <- file.path(.s1_pack_dir(study), "index.json")
    if (!file.exists(ip) && identical(study$transport$kind, "http")) {
        tryCatch(writeBin(.s1_get_ref(study$transport, "index.json")$body, ip), error = function(e) NULL)
    }
    mp <- file.path(root, "m", paste0(study$gse, ".json"))
    dir.create(dirname(mp), recursive = TRUE, showWarnings = FALSE)
    if (!file.exists(mp)) writeLines(jsonlite::toJSON(study$doc, auto_unbox = TRUE, null = "null", digits = NA), mp)
    mp
}


# ---------------------------------------------------------------------------
# Entry points used by load() / read_singlet()
# ---------------------------------------------------------------------------

# A per-sample study by accession; a study with no .s1 manifest stops with a message naming legacy = TRUE
# (SINGLET_LEGACY_FALLBACK=1 falls back to the legacy bundle instead).
.s1_load_gse <- function(gse, gsms = NULL, cache_dir = NULL, legacy = NULL, modalities = NULL, labels = list()) {
    study <- tryCatch(singlet_study(gse, cache_dir = cache_dir), singlet_s1_not_found = function(e) e)
    if (inherits(study, "condition")) {
        if (is.null(legacy) && .s1_legacy_fallback()) {
            if (nzchar(Sys.getenv("SINGLET_LEGACY_FALLBACK"))) {
                warning(sprintf("%s; using the legacy bundle (SINGLET_LEGACY_FALLBACK)", conditionMessage(study)), call. = FALSE)
            }
            return(.legacy_modalities(.read_singlet_bundle(.singlet_fetch_bundle(gse, cache_dir = cache_dir), gsms = gsms),
                                      modalities))
        }
        stop(study)
    }
    do.call(.s1_read_study, c(list(study, gsms, modalities = modalities), labels))
}

# modalities= on a legacy bundle: it has been downloaded whole already, so only the gene assays are subset (with a warning);
# the feature-level modalities need the .s1 reader and are left out with a warning, never an error.
.legacy_modalities <- function(sce, modalities) {
    if (is.null(modalities)) return(sce)
    mods <- .s1_normalize_modalities(modalities)
    warning("this study is read from its legacy .singlet bundle, which is downloaded whole; modalities= only subsets the result",
            call. = FALSE)
    drop <- setdiff(intersect(SummarizedExperiment::assayNames(sce), .s1_gene_modalities), mods)
    for (a in drop) SummarizedExperiment::assay(sce, a) <- NULL
    other <- setdiff(mods, .s1_gene_modalities)
    if (length(other) > 0L) {
        warning(sprintf("%s: not read from a legacy bundle (use the per-sample study, or read_singlet() members)", paste(other, collapse = ", ")),
                call. = FALSE)
    }
    sce
}

# A local manifest, .../s1/m/<GSE>.json (its objects are read from the same tree; DUAL: older v2/m/ trees too).
.s1_load_manifest <- function(path, gsms = NULL, modalities = NULL, labels = list()) {
    p <- normalizePath(path, winslash = "/", mustWork = TRUE)
    if (basename(dirname(p)) != "m" || !(basename(dirname(dirname(p))) %in% c("s1", "store", "v2"))) {
        stop(sprintf("%s is not under a s1/m/ directory", path), call. = FALSE)
    }
    root <- dirname(dirname(dirname(p)))
    gse <- sub("\\.json$", "", basename(p), ignore.case = TRUE)
    do.call(.s1_read_study, c(list(singlet_study(gse, base = root, force = TRUE), gsms, modalities = modalities), labels))
}
