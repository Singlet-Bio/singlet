# SPDX-License-Identifier: MIT
# User-facing entry points: load(), find(), find_load().
#
# Design principle: R users only ever handle `.singlet` bundles and GEO
# accession strings (GSE.../GSM...). The lower-level `.1pz` codec is never
# required in the documented public API.


# ---------------------------------------------------------------------------
# Internal: configuration resolved from environment variables.
# ---------------------------------------------------------------------------

# Base URL serving .singlet bundles at <base>/<GSE>/<GSE>.singlet
.singlet_data_base <- function() {
    Sys.getenv("SINGLET_DATA_BASE", unset = "https://data.singlet.bio/data")
}

# REST API base for natural-language search.
.singlet_api_base <- function() {
    Sys.getenv("SINGLET_API_BASE", unset = "https://singlet.bio/api")
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

.is_accession <- function(x) {
    grepl("^GS[EM][0-9]+$", x)
}

.gse_of <- function(accession) {
    # GSE accessions map directly; GSM accessions are not currently a valid
    # bundle key on data.singlet.bio (bundles are per-GSE), so we surface a
    # clear error there.
    if (grepl("^GSE", accession)) {
        return(accession)
    }
    stop(sprintf(
        "GSM-level download is not supported directly: %s. Bundles are ",
        accession),
        "distributed per Series (GSE). Pass the parent GSE accession instead.")
}


# ---------------------------------------------------------------------------
# Internal: ensure a .singlet bundle for an accession is on disk, return path.
# ---------------------------------------------------------------------------
.singlet_fetch_bundle <- function(accession, cache_dir = NULL) {
    gse <- .gse_of(accession)
    cache <- .singlet_cache_dir(cache_dir)
    dest <- file.path(cache, paste0(gse, ".singlet"))
    if (file.exists(dest) && file.size(dest) > 0L) {
        return(dest)
    }
    url <- sprintf("%s/%s/%s.singlet", .singlet_data_base(), gse, gse)
    tmp <- paste0(dest, ".part")
    old_to <- getOption("timeout")
    on.exit(options(timeout = old_to), add = TRUE)
    options(timeout = max(600, old_to))
    status <- tryCatch(
        utils::download.file(url, tmp, mode = "wb", quiet = TRUE),
        error = function(e) {
            unlink(tmp)
            stop(sprintf("failed to download bundle for %s from %s: %s",
                         gse, url, conditionMessage(e)))
        }
    )
    if (!file.exists(tmp) || file.size(tmp) == 0L) {
        unlink(tmp)
        stop(sprintf("download produced an empty file for %s (%s)", gse, url))
    }
    file.rename(tmp, dest)
    dest
}


# ---------------------------------------------------------------------------
# load — the primary user-facing entry point.
# ---------------------------------------------------------------------------

#' Load Singlet data by accession or bundle path
#'
#' The primary entry point for working with the Singlet atlas from R. Accepts
#' one or more inputs that may be GEO accessions (\code{"GSE..."}), local
#' \code{.singlet} bundle paths, or a mix of both, and returns a single
#' combined object. GEO accessions are downloaded (and cached) from
#' \url{https://singlet.bio} on demand; local paths are read directly.
#' Multiple inputs are combined column-wise on their shared gene axis.
#'
#' @param x A character vector of GEO Series accessions (\code{"GSE..."})
#'   and/or paths to local \code{.singlet} bundles. Length one or more.
#' @param as Output class: \code{"sce"} (default) for a
#'   \code{SingleCellExperiment}, or \code{"seurat"} for a \code{Seurat}
#'   object.
#' @param cache_dir Directory in which to cache downloaded bundles. Defaults
#'   to \code{tools::R_user_dir("singlet", "cache")} (override with the
#'   \code{SINGLET_CACHE_DIR} environment variable). Existing bundles are
#'   reused rather than re-downloaded.
#' @return A combined \code{SingleCellExperiment} (when \code{as = "sce"}) or
#'   \code{Seurat} object (when \code{as = "seurat"}) spanning all inputs.
#'
#' @details
#' Downloads honor two environment variables: \code{SINGLET_DATA_BASE} (the
#' base URL for bundles, default \code{https://data.singlet.bio/data}) and
#' \code{SINGLET_CACHE_DIR} (the on-disk cache location).
#'
#' @examples
#' \dontrun{
#' # One study by accession (downloaded + cached automatically)
#' sce <- load("GSE149298")
#'
#' # Combine several studies into one object
#' sce <- load(c("GSE149298", "GSE116256"))
#'
#' # A local bundle, returned as a Seurat object
#' obj <- load("/data/GSE149298.singlet", as = "seurat")
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

    paths <- vapply(x, function(item) {
        if (.is_accession(item)) {
            .singlet_fetch_bundle(item, cache_dir = cache_dir)
        } else {
            p <- path.expand(item)
            if (!file.exists(p)) {
                stop(sprintf(
                    "input is neither a GSE/GSM accession nor an existing file: %s",
                    item))
            }
            p
        }
    }, character(1))

    sces <- lapply(unname(paths), read_singlet)
    sce <- if (length(sces) == 1L) sces[[1]] else .combine_sces(sces)

    if (as == "seurat") {
        return(.sce_to_seurat(sce))
    }
    sce
}


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
    do.call(SingleCellExperiment::cbind, aligned)
}


# ---------------------------------------------------------------------------
# Internal: convert a SingleCellExperiment to a Seurat object.
# ---------------------------------------------------------------------------
.sce_to_seurat <- function(sce) {
    if (!requireNamespace("Seurat", quietly = TRUE)) {
        stop("as = \"seurat\" requires the Seurat package. ",
             "Install with `install.packages('Seurat')`.")
    }
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
#'
#' @param query Free-text search string, e.g.
#'   \code{"T cells from pediatric AML"}.
#' @param level Accession granularity to return: \code{"gsm"} (default, one
#'   accession per sample) or \code{"gse"} (one per Series).
#' @param limit Maximum number of accessions to return. Default \code{50}.
#' @return A character vector of matching accessions (possibly empty).
#'
#' @details
#' The API base URL is taken from the \code{SINGLET_API_BASE} environment
#' variable when set, otherwise \code{https://singlet.bio/api}. Requires
#' network access.
#'
#' @examples
#' \dontrun{
#' hits <- find("T cells from pediatric AML", level = "gse")
#' sce <- load(hits[1:3])
#' }
#'
#' @seealso \code{\link{load}}, \code{\link{find_load}}
#' @export
find <- function(query, level = c("gsm", "gse"), limit = 50L) {
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


# ---------------------------------------------------------------------------
# Internal: lightweight HTTP GET returning the response body as a string.
#
# Uses the curl package when available (more robust) and falls back to a
# base-R url()/readLines() connection so the package has no hard network
# dependency.
# ---------------------------------------------------------------------------
.singlet_http_get <- function(url) {
    if (requireNamespace("curl", quietly = TRUE)) {
        return(tryCatch(
            rawToChar(curl::curl_fetch_memory(url)$content),
            error = function(e) {
                stop(sprintf("search request failed (%s): %s",
                             url, conditionMessage(e)))
            }
        ))
    }
    con <- tryCatch(url(url, open = "rb"),
                    error = function(e) {
                        stop(sprintf("search request failed (%s): %s",
                                     url, conditionMessage(e)))
                    })
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
#' sce <- find_load("microglia from Alzheimer's brain", limit = 5)
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
