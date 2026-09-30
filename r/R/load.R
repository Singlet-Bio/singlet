# SPDX-License-Identifier: MIT
# User-facing entry points: load(), find(), find_load(), download(), and
# their singlet_*() aliases.
#
# Design principle: R users only ever handle `.singlet` bundles and GEO
# accession strings (GSE.../GSM...). The lower-level `.1pz` codec is never
# required in the documented public API.
#
# load() and find() mask base::load() and utils::find() once the package is
# attached. The singlet_*() aliases are the same functions under names that
# mask nothing.


# ---------------------------------------------------------------------------
# Internal: configuration resolved from environment variables.
# ---------------------------------------------------------------------------

# Host serving .singlet bundles. Files live at
# <base>/data/<GSE>/<GSE>.singlet, matching the Python client, where
# SINGLET_DATA_BASE also names the host. A base that already ends in /data
# (what this package expected before 1.1.0) is accepted too.
.singlet_data_base <- function() {
    base <- Sys.getenv("SINGLET_DATA_BASE", unset = "")
    if (!nzchar(base)) {
        base <- "https://data.singlet.bio"
    }
    base <- sub("/+$", "", base)
    sub("/data$", "", base)
}

# Public URL of a Series' bundle.
.singlet_bundle_url <- function(gse) {
    sprintf("%s/data/%s/%s.singlet", .singlet_data_base(), gse, gse)
}

# REST API base for natural-language search and GSM -> GSE lookups.
.singlet_api_base <- function() {
    Sys.getenv("SINGLET_API_BASE", unset = "https://singlet.bio/api")
}


# Package version string for the User-Agent header. The search API recognises
# the "singlet-r/" prefix and applies the client-library rate limit.
.singlet_user_agent <- function() {
    v <- tryCatch(as.character(utils::packageVersion("singlet")),
                  error = function(e) "0")
    paste0("singlet-r/", v)
}

.singlet_key_env <- new.env(parent = emptyenv())

.singlet_api_key <- function() {
    k <- .singlet_key_env$key
    if (!is.null(k) && nzchar(k)) return(k)
    trimws(Sys.getenv("SINGLET_API_KEY", unset = ""))
}

#' Set the API key used for natural-language search
#'
#' Natural-language search (\code{\link{find}}) is AI-interpreted and
#' rate-limited per client. Heavy use needs an API key created at
#' \url{https://singlet.bio/account}. The key is sent as a bearer token on
#' search requests only; downloads and \code{\link{load}} never need one.
#' When no key has been set with this function, the \code{SINGLET_API_KEY}
#' environment variable is used.
#'
#' @param key Character API key (\code{"sk_live_..."}), or \code{NULL} to
#'   clear a previously set key.
#' @return The previously set key (invisibly), or \code{NULL}.
#' @examples
#' \dontrun{
#' set_api_key("sk_live_...")
#' hits <- singlet_find("microglia in the aging mouse brain")
#' }
#' @seealso \code{\link{find}}
#' @export
set_api_key <- function(key) {
    old <- .singlet_key_env$key
    if (is.null(key) || !nzchar(key)) {
        .singlet_key_env$key <- NULL
    } else {
        .singlet_key_env$key <- trimws(as.character(key)[1])
    }
    invisible(old)
}

# Local cache directory for downloaded bundles.
.singlet_cache_dir <- function(cache_dir = NULL) {
    if (!is.null(cache_dir)) {
        d <- path.expand(cache_dir)
    } else {
        env <- Sys.getenv("SINGLET_CACHE_DIR", unset = "")
        if (nzchar(env)) {
            d <- path.expand(env)
        } else {
            d <- tryCatch(
                tools::R_user_dir("singlet", "cache"),
                error = function(e) path.expand(file.path("~", ".singlet", "cache"))
            )
        }
    }
    if (!dir.exists(d)) {
        dir.create(d, recursive = TRUE, showWarnings = FALSE)
    }
    d
}


# ---------------------------------------------------------------------------
# Internal: accessions.
# ---------------------------------------------------------------------------

.is_accession <- function(x) {
    grepl("^GS[EM][0-9]+$", x)
}

# Normalise one input to an accession, or NA when it is not one. Lower-case
# or padded accessions ("gse138867 ") are accepted unless a file of that
# name exists, in which case the input is a path.
.as_accession <- function(x) {
    if (.is_accession(x)) {
        return(x)
    }
    y <- toupper(trimws(x))
    if (.is_accession(y) && !file.exists(path.expand(x))) {
        return(y)
    }
    NA_character_
}

# GSM -> parent GSE lookups, cached for the session so a vector of samples
# from one study costs one request.
.singlet_gsm_parents <- new.env(parent = emptyenv())

# The same mapping, kept on disk next to the bundle cache so a sample whose
# study is already cached loads without the network in a later session.
# A tab-separated file of "GSM<TAB>GSE" lines; the last line for a GSM wins.
.gsm_index_path <- function(cache) {
    file.path(cache, "gsm_parents.tsv")
}

.gsm_index_lookup <- function(cache, gsm) {
    if (is.null(cache)) {
        return(NULL)
    }
    idx <- .gsm_index_path(cache)
    if (!file.exists(idx)) {
        return(NULL)
    }
    lines <- tryCatch(readLines(idx, warn = FALSE),
                      error = function(e) character(0))
    hits <- lines[startsWith(lines, paste0(gsm, "\t"))]
    if (length(hits) == 0L) {
        return(NULL)
    }
    gse <- trimws(sub("^[^\t]*\t", "", hits[[length(hits)]]))
    if (grepl("^GSE[0-9]+$", gse)) gse else NULL
}

# Best effort: a cache directory that cannot be written only costs the
# offline shortcut, never the load itself.
.gsm_index_record <- function(cache, gsm, gse) {
    if (is.null(cache)) {
        return(invisible(FALSE))
    }
    ok <- tryCatch({
        cat(gsm, "\t", gse, "\n", sep = "", file = .gsm_index_path(cache),
            append = TRUE)
        TRUE
    }, error = function(e) FALSE, warning = function(w) FALSE)
    invisible(ok)
}

# Last resort when the API cannot be reached: a cached bundle that holds the
# sample. Only the zip directory of each bundle is read.
.gsm_in_cached_bundles <- function(cache, gsm) {
    if (is.null(cache) || !dir.exists(cache)) {
        return(NULL)
    }
    bundles <- list.files(cache, pattern = "^GSE[0-9]+\\.singlet$",
                          full.names = TRUE)
    for (b in bundles) {
        ids <- tryCatch(suppressWarnings(.bundle_gsm_ids(b)),
                        error = function(e) character(0))
        if (gsm %in% ids) {
            return(sub("\\.singlet$", "", basename(b)))
        }
    }
    NULL
}

# Parent series from either API response shape:
#   GET <api>/gsm/<GSM>     {"sample": {"gsm_id", "gse_id", ...},
#                            "series": {"id", ...}, "siblings": [...]}
#   GET <api>/gsm?q=<GSM>   {"data": [{"gsm_id", "gse_id", ...}, ...]}
# A listing row is only trusted when it names this exact sample.
.gse_from_payload <- function(payload, gsm) {
    if (!is.list(payload)) {
        return(NULL)
    }
    candidates <- list(
        .json_field(payload, c("sample", "gse_id")),
        .json_field(payload, c("series", "id")),
        .json_field(payload, "gse_id")
    )
    rows <- .json_field(payload, "data")
    if (is.list(rows) && is.null(names(rows))) {
        for (row in rows) {
            if (identical(.json_field(row, "gsm_id"), gsm)) {
                candidates <- c(candidates, list(.json_field(row, "gse_id")))
            }
        }
    }
    for (v in candidates) {
        if (is.character(v) && length(v) == 1L && grepl("^GSE[0-9]+$", v)) {
            return(v)
        }
    }
    NULL
}

# Resolve a GSM accession to its parent GSE. Lookups made before are reused
# from the session cache, then from the on-disk index in `cache` (the bundle
# cache directory; NULL skips everything on disk). Otherwise the public API
# is asked: the sample-detail endpoint first, then the listing endpoint (the
# one the Python client uses), so either one being unavailable is
# survivable. If both fail, a cached bundle that holds the sample answers.
.resolve_gsm_parent <- function(gsm, cache = NULL) {
    hit <- .singlet_gsm_parents[[gsm]]
    if (!is.null(hit)) {
        return(hit)
    }
    hit <- .gsm_index_lookup(cache, gsm)
    if (!is.null(hit)) {
        assign(gsm, hit, envir = .singlet_gsm_parents)
        message(sprintf("%s belongs to %s; using the %s bundle", gsm, hit, hit))
        return(hit)
    }
    base <- sub("/+$", "", .singlet_api_base())
    enc <- utils::URLencode(gsm, reserved = TRUE)
    urls <- c(
        sprintf("%s/gsm/%s", base, enc),
        sprintf("%s/gsm?q=%s&limit=5", base, enc)
    )
    problems <- character(0)
    gse <- NULL
    for (u in urls) {
        payload <- tryCatch(
            jsonlite::fromJSON(
                .singlet_http_get(u, auth = FALSE, what = "sample lookup"),
                simplifyVector = FALSE
            ),
            error = function(e) {
                problems <<- c(problems, conditionMessage(e))
                NULL
            }
        )
        gse <- .gse_from_payload(payload, gsm)
        if (!is.null(gse)) break
    }
    if (is.null(gse)) {
        gse <- .gsm_in_cached_bundles(cache, gsm)
        if (!is.null(gse)) {
            message(sprintf(
                "sample lookup via %s failed; %s is in the cached %s bundle",
                base, gsm, gse))
        }
    }
    if (is.null(gse)) {
        detail <- if (length(problems) > 0L) {
            paste0(" (", paste(problems, collapse = "; "), ")")
        } else {
            ""
        }
        stop(sprintf(
            "could not resolve sample %s to its parent series via %s%s. Pass the parent GSE accession instead.",
            gsm, base, detail
        ), call. = FALSE)
    }
    assign(gsm, gse, envir = .singlet_gsm_parents)
    .gsm_index_record(cache, gsm, gse)
    message(sprintf("%s belongs to %s; using the %s bundle", gsm, gse, gse))
    gse
}

# Bundles are per Series: a GSE maps to itself, a GSM to its parent. `cache`
# is the bundle cache directory, where GSM -> GSE lookups are remembered.
.gse_of <- function(accession, cache = NULL) {
    if (grepl("^GSE[0-9]+$", accession)) {
        return(accession)
    }
    if (grepl("^GSM[0-9]+$", accession)) {
        return(.resolve_gsm_parent(accession, cache = cache))
    }
    stop(sprintf("not a GEO accession (GSE... or GSM...): %s", accession),
         call. = FALSE)
}


# ---------------------------------------------------------------------------
# Internal: ensure a .singlet bundle for an accession is on disk, return path.
# ---------------------------------------------------------------------------
.singlet_fetch_bundle <- function(accession, cache_dir = NULL) {
    cache <- .singlet_cache_dir(cache_dir)
    gse <- .gse_of(accession, cache = cache)
    dest <- file.path(cache, paste0(gse, ".singlet"))
    if (file.exists(dest) && file.size(dest) > 0L) {
        return(dest)
    }
    url <- .singlet_bundle_url(gse)
    tmp <- paste0(dest, ".part")
    old_to <- getOption("timeout")
    on.exit(options(timeout = old_to), add = TRUE)
    options(timeout = max(600, old_to))
    status <- tryCatch(
        utils::download.file(url, tmp, mode = "wb", quiet = TRUE),
        error = function(e) {
            unlink(tmp)
            stop(sprintf(
                "failed to download bundle for %s from %s: %s. The study may have no published bundle; check it at https://singlet.bio",
                gse, url, conditionMessage(e)), call. = FALSE)
        }
    )
    if (!identical(as.integer(status), 0L) ||
        !file.exists(tmp) || file.size(tmp) == 0L) {
        unlink(tmp)
        stop(sprintf("download produced an empty file for %s (%s)", gse, url))
    }
    if (!file.rename(tmp, dest)) {
        unlink(tmp)
        stop(sprintf("could not move the downloaded bundle into the cache: %s",
                     dest))
    }
    dest
}


#' Download a `.singlet` bundle and return its local path
#'
#' Fetches a study's bundle from \url{https://data.singlet.bio} into the
#' local cache (or reuses it if already there) and returns the path, without
#' reading it. Use this when you want to reach individual modalities with
#' \code{\link{singlet_read}} or \code{\link{singlet_raw_counts}} rather than
#' load the whole study. \code{singlet_download()} is the same function.
#'
#' @param accession A GEO Series accession (\code{"GSE..."}). A sample
#'   accession (\code{"GSM..."}) is resolved to its parent Series through the
#'   Singlet API, and the whole Series bundle is downloaded. The mapping is
#'   remembered in the cache directory, so a sample whose Series is already
#'   cached needs no network in later sessions.
#' @param cache_dir Directory in which to cache the bundle. Defaults to
#'   \code{tools::R_user_dir("singlet", "cache")}, overridable with the
#'   \code{SINGLET_CACHE_DIR} environment variable.
#' @return The path to the local `.singlet` file.
#'
#' @details
#' Bundles are fetched from \code{<base>/data/<GSE>/<GSE>.singlet}, where
#' \code{<base>} is the \code{SINGLET_DATA_BASE} environment variable
#' (default \code{https://data.singlet.bio}; a value ending in \code{/data}
#' is accepted). \code{GSM} accessions are resolved with the API named by
#' \code{SINGLET_API_BASE} (default \code{https://singlet.bio/api}); each
#' answer is saved in \code{gsm_parents.tsv} in the cache directory, and when
#' the API cannot be reached the cached bundles are searched for the sample.
#'
#' @examples
#' \dontrun{
#' path <- singlet_download("GSE138867")
#' singlet_modalities(path)
#'
#' # A sample accession fetches its parent study's bundle
#' identical(singlet_download("GSM4120733"), path)
#' }
#'
#' @seealso \code{\link{load}}, \code{\link{singlet_modalities}}
#' @export
download <- function(accession, cache_dir = NULL) {
    acc <- .as_accession(as.character(accession)[[1L]])
    if (is.na(acc)) {
        stop(sprintf("not a GEO accession (GSE... or GSM...): %s",
                     as.character(accession)[[1L]]), call. = FALSE)
    }
    .singlet_fetch_bundle(acc, cache_dir = cache_dir)
}

#' @rdname download
#' @export
singlet_download <- download


# ---------------------------------------------------------------------------
# load — the primary user-facing entry point.
# ---------------------------------------------------------------------------

#' Load Singlet data by accession or bundle path
#'
#' The primary entry point for working with the Singlet atlas from R. Accepts
#' one or more inputs that may be GEO accessions (\code{"GSE..."} or
#' \code{"GSM..."}), local \code{.singlet} bundle paths, or a mix of these,
#' and returns a single combined object. Accessions are downloaded (and
#' cached) from \url{https://data.singlet.bio} on demand; local paths are read
#' directly. Multiple inputs are combined column-wise on their shared gene
#' axis. \code{singlet_load()} is the same function.
#'
#' @param x A character vector of GEO Series accessions (\code{"GSE..."}),
#'   GEO Sample accessions (\code{"GSM..."}) and/or paths to local
#'   \code{.singlet} bundles. Length one or more. A sample accession is
#'   resolved to its parent Series through the Singlet API; that Series'
#'   bundle is downloaded and only the sample's cells are kept.
#' @param as Output class: \code{"sce"} (default) for a
#'   \code{SingleCellExperiment}, or \code{"seurat"} for a \code{Seurat}
#'   object.
#' @param cache_dir Directory in which to cache downloaded bundles. Defaults
#'   to \code{tools::R_user_dir("singlet", "cache")} (override with the
#'   \code{SINGLET_CACHE_DIR} environment variable). Existing bundles are
#'   reused rather than re-downloaded, and \code{GSM} lookups are
#'   remembered there, so a sample of an already cached study loads offline.
#' @return A combined \code{SingleCellExperiment} (when \code{as = "sce"}) or
#'   \code{Seurat} object (when \code{as = "seurat"}) spanning all inputs.
#'   Samples with no usable cells (for example an empty count matrix left by
#'   a failed pipeline run) are skipped with a warning and listed, with the
#'   reason, in \code{metadata(sce)$skipped_samples}
#'   (\code{obj@misc$skipped_samples} for Seurat).
#'
#' @details
#' Downloads honor three environment variables: \code{SINGLET_DATA_BASE}
#' (the host serving bundles, default \code{https://data.singlet.bio};
#' bundles are fetched from \code{<base>/data/<GSE>/<GSE>.singlet}),
#' \code{SINGLET_API_BASE} (used to resolve \code{GSM} accessions, default
#' \code{https://singlet.bio/api}) and \code{SINGLET_CACHE_DIR} (the on-disk
#' cache location).
#'
#' \code{load()} masks \code{base::load()} once the package is attached;
#' \code{singlet_load()} is the same function under a name that masks
#' nothing.
#'
#' @examples
#' \dontrun{
#' # One study by accession (downloaded + cached automatically)
#' sce <- singlet_load("GSE138867")
#'
#' # One sample: resolves to GSE138867 and keeps only GSM4120733's cells
#' sce <- singlet_load("GSM4120733")
#'
#' # Combine several studies into one object
#' sce <- singlet_load(c("GSE138867", "GSE146974"))
#'
#' # Samples that were skipped for having no usable cells, and why
#' S4Vectors::metadata(sce)$skipped_samples
#'
#' # A local bundle, returned as a Seurat object
#' obj <- singlet_load("GSE138867.singlet", as = "seurat")
#' }
#'
#' @seealso \code{\link{read_singlet}}, \code{\link{find}},
#'   \code{\link{find_load}}
#' @export
load <- function(x, as = c("sce", "seurat"), cache_dir = NULL) {
    as <- match.arg(as)
    x <- as.character(x)
    if (length(x) == 0L) {
        stop("load: `x` must contain at least one accession or bundle path")
    }

    # Classify every input before touching the network, so a typo in the
    # last element does not cost a download of the first.
    accessions <- vapply(x, .as_accession, character(1), USE.NAMES = FALSE)
    for (i in which(is.na(accessions))) {
        if (!file.exists(path.expand(x[[i]]))) {
            stop(sprintf(
                "input is neither a GSE/GSM accession nor an existing file: %s",
                x[[i]]))
        }
    }
    .require_sce("load()")
    if (as == "seurat") {
        .require_seurat("load(as = \"seurat\")")
    }

    # One entry per input: the bundle on disk plus, for a GSM accession, the
    # sample to keep (NA keeps the whole study).
    targets <- lapply(seq_along(x), function(i) {
        acc <- accessions[[i]]
        if (is.na(acc)) {
            return(list(path = path.expand(x[[i]]), gsm = NA_character_))
        }
        list(path = .singlet_fetch_bundle(acc, cache_dir = cache_dir),
             gsm = if (startsWith(acc, "GSM")) acc else NA_character_)
    })

    # Read each bundle once, however many of its samples were asked for.
    paths <- vapply(targets, function(t) t$path, character(1))
    sces <- lapply(unique(paths), function(p) {
        gsms <- vapply(targets[paths == p], function(t) t$gsm, character(1))
        .read_singlet_bundle(p, gsms = if (anyNA(gsms)) NULL else unique(gsms))
    })
    sce <- if (length(sces) == 1L) sces[[1]] else .combine_sces(sces)

    if (as == "seurat") {
        return(.sce_to_seurat(sce))
    }
    sce
}

#' @rdname load
#' @export
singlet_load <- load


# ---------------------------------------------------------------------------
# Internal: combine multiple SCEs on their shared gene axis.
# ---------------------------------------------------------------------------
.combine_sces <- function(sces) {
    # Restrict every SCE to the intersection of genes, in a common order,
    # then cbind. Bundles share the same reference build so the gene axis is
    # typically identical, but we intersect defensively.
    gene_sets <- lapply(sces, rownames)
    common <- Reduce(intersect, gene_sets)
    if (length(common) == 0L) {
        stop("cannot combine bundles: no genes in common across inputs")
    }
    aligned <- lapply(sces, function(s) s[common, , drop = FALSE])
    out <- do.call(SingleCellExperiment::cbind, aligned)

    # cbind() concatenates metadata lists, which would leave one
    # skipped_samples entry per input. Keep a single combined table.
    skipped <- do.call(rbind, lapply(sces, function(s) {
        S4Vectors::metadata(s)[["skipped_samples"]]
    }))
    md <- S4Vectors::metadata(out)
    md <- md[names(md) != "skipped_samples"]
    md[["skipped_samples"]] <- skipped
    S4Vectors::metadata(out) <- md
    out
}


# ---------------------------------------------------------------------------
# Internal: convert a SingleCellExperiment to a Seurat object.
# ---------------------------------------------------------------------------
.sce_to_seurat <- function(sce) {
    .require_seurat("load(as = \"seurat\")")
    counts <- SummarizedExperiment::assay(sce, "counts")
    obj <- Seurat::CreateSeuratObject(
        counts = counts,
        assay = "RNA",
        min.cells = 0,
        min.features = 0
    )
    cd <- as.data.frame(SummarizedExperiment::colData(sce),
                        check.names = FALSE)
    if (ncol(cd) > 0L) {
        obj <- Seurat::AddMetaData(obj, metadata = cd)
    }
    md <- S4Vectors::metadata(sce)
    obj@misc$manifest <- md$manifest
    obj@misc$study_meta <- md$study_meta
    obj@misc$skipped_samples <- md$skipped_samples
    obj
}


# ---------------------------------------------------------------------------
# find — natural-language search over the atlas.
# ---------------------------------------------------------------------------

#' Search the Singlet atlas with natural language
#'
#' Query the Singlet atlas with a free-text description and receive matching
#' GEO accessions. The query is sent to the Singlet search API
#' (\url{https://singlet.bio/api}); pair the result with \code{\link{load}}
#' (or use \code{\link{find_load}}) to pull the matching data.
#' \code{singlet_find()} is the same function.
#'
#' @param query Free-text search string, e.g.
#'   \code{"T cells from pediatric AML"}.
#' @param level Accession granularity to return: \code{"gse"} (default, one
#'   accession per Series) or \code{"gsm"} (one per sample). Both kinds can
#'   be passed straight to \code{\link{load}}: a \code{GSM} accession is
#'   resolved to its parent Series and only that sample's cells are kept.
#' @param limit Maximum number of accessions to return. Default \code{50}.
#' @return A character vector of matching accessions (possibly empty).
#'
#' @details
#' The API base URL is taken from the \code{SINGLET_API_BASE} environment
#' variable when set, otherwise \code{https://singlet.bio/api}. Requires
#' network access. \code{find()} masks \code{utils::find()} once the package
#' is attached; \code{singlet_find()} masks nothing.
#'
#' @examples
#' \dontrun{
#' hits <- singlet_find("T cells from pediatric AML")
#' sce <- singlet_load(hits[1:3])
#'
#' # One accession per sample instead of per study
#' samples <- singlet_find("human PBMC from smokers", level = "gsm")
#' }
#'
#' @seealso \code{\link{load}}, \code{\link{find_load}}
#' @export
find <- function(query, level = c("gse", "gsm"), limit = 50L) {
    level <- match.arg(level)
    query <- as.character(query)[1]
    if (!nzchar(query)) {
        stop("find: `query` must be a non-empty string")
    }
    url <- sprintf(
        "%s/nl-search?q=%s&level=%s&limit=%d",
        .singlet_api_base(),
        utils::URLencode(query, reserved = TRUE),
        level,
        as.integer(limit)
    )
    txt <- .singlet_http_get(url)
    parsed <- jsonlite::fromJSON(txt, simplifyVector = TRUE)
    accs <- parsed$accessions
    if (is.null(accs)) {
        return(character(0))
    }
    as.character(accs)
}

#' @rdname find
#' @export
singlet_find <- find


# ---------------------------------------------------------------------------
# Internal: lightweight HTTP GET returning the response body as a string.
#
# Uses the curl package when available (more robust, and the only way to
# send the API key) and falls back to a base-R url()/readLines() connection
# so the package has no hard network dependency. `auth = FALSE` is for
# catalog lookups, which need no key and get no search-specific messages.
# ---------------------------------------------------------------------------
.singlet_http_get <- function(url, auth = TRUE, what = "search request",
                              timeout = 120) {
    key <- if (auth) .singlet_api_key() else ""
    if (requireNamespace("curl", quietly = TRUE)) {
        h <- curl::new_handle()
        curl::handle_setopt(h, timeout = timeout)
        headers <- c(
            "User-Agent" = .singlet_user_agent(),
            "Accept" = "application/json"
        )
        if (nzchar(key)) {
            headers <- c(headers, "Authorization" = paste("Bearer", key))
        }
        do.call(curl::handle_setheaders, c(list(h), as.list(headers)))
        res <- tryCatch(
            curl::curl_fetch_memory(url, handle = h),
            error = function(e) {
                stop(sprintf("%s failed (%s): %s",
                             what, url, conditionMessage(e)), call. = FALSE)
            }
        )
        if (auth && res$status_code == 429L) {
            stop(paste0(
                "natural-language search limit reached. Create an API key at ",
                "https://singlet.bio/account and set SINGLET_API_KEY ",
                "(or call singlet::set_api_key()) for a higher limit."
            ), call. = FALSE)
        }
        if (auth && res$status_code %in% c(401L, 403L)) {
            stop(sprintf(
                "search request was rejected (HTTP %d); check SINGLET_API_KEY",
                res$status_code
            ), call. = FALSE)
        }
        if (res$status_code >= 400L) {
            stop(sprintf("%s failed (%s): HTTP %d",
                         what, url, res$status_code), call. = FALSE)
        }
        return(rawToChar(res$content))
    }
    if (nzchar(key)) {
        warning("SINGLET_API_KEY is set but the 'curl' package is not ",
                "installed; the key cannot be sent. install.packages(\"curl\")")
    }
    old <- options(HTTPUserAgent = .singlet_user_agent())
    on.exit(options(old), add = TRUE)
    # base::url() reports the HTTP status as a warning before failing; fold
    # it into the error instead of leaving a stray warning behind.
    notes <- character(0)
    con <- withCallingHandlers(
        tryCatch(base::url(url, open = "rb"), error = function(e) {
            stop(sprintf("%s failed (%s): %s", what, url,
                         paste(c(notes, conditionMessage(e)), collapse = "; ")),
                 call. = FALSE)
        }),
        warning = function(w) {
            notes <<- c(notes, conditionMessage(w))
            invokeRestart("muffleWarning")
        }
    )
    on.exit(close(con), add = TRUE)
    raw <- readLines(con, warn = FALSE)
    paste(raw, collapse = "\n")
}


# ---------------------------------------------------------------------------
# find_load — convenience: search then load the matches.
# ---------------------------------------------------------------------------

#' Search and load in one step
#'
#' Convenience wrapper that runs \code{\link{find}} and immediately passes
#' the matching accessions to \code{\link{load}}.
#' \code{singlet_find_load()} is the same function.
#'
#' @param query Free-text search string (see \code{\link{find}}).
#' @param as Output class, \code{"sce"} (default) or \code{"seurat"}.
#' @param level Accession granularity, \code{"gse"} (default here, since
#'   whole Series combine cleanly) or \code{"gsm"}.
#' @param limit Maximum number of accessions to load. Default \code{10}.
#' @param cache_dir Bundle cache directory (see \code{\link{load}}).
#' @return A combined \code{SingleCellExperiment} or \code{Seurat} object, or
#'   \code{NULL} (invisibly) if the search returns no matches.
#'
#' @examples
#' \dontrun{
#' sce <- singlet_find_load("microglia from Alzheimer's brain", limit = 5)
#' }
#'
#' @seealso \code{\link{find}}, \code{\link{load}}
#' @export
find_load <- function(query, as = c("sce", "seurat"),
                      level = c("gse", "gsm"), limit = 10L,
                      cache_dir = NULL) {
    as <- match.arg(as)
    level <- match.arg(level)
    hits <- find(query, level = level, limit = limit)
    if (length(hits) == 0L) {
        message("find_load: no matches for query: ", query)
        return(invisible(NULL))
    }
    load(hits, as = as, cache_dir = cache_dir)
}

#' @rdname find_load
#' @export
singlet_find_load <- find_load
