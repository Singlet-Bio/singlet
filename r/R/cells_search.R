# Atlas-wide cell search: the singlet.bio cell index (GET /api/cells, /api/cells/slice), the R counterpart of the MCP tools
# search_cells / get_cells_page and of singlet.s1.atlas.gather() in Python.

# GET <api>/<path>?<params> and parse the JSON answer. Cell search needs no key; one set with set_api_key() is sent.
.singlet_api_json <- function(path, params = list(), what = "cell search") {
    params <- Filter(Negate(is.null), params)
    q <- if (length(params)) {
        paste0("?", paste(vapply(names(params), function(k) {
            paste0(k, "=", utils::URLencode(as.character(params[[k]]), reserved = TRUE))
        }, ""), collapse = "&"))
    } else ""
    url <- paste0(sub("/+$", "", .singlet_api_base()), "/", path, q)
    jsonlite::fromJSON(.singlet_http_get(url, auth = FALSE, what = what), simplifyVector = FALSE)
}

.singlet_where_json <- function(where) {
    if (is.null(where)) return(NULL)
    if (is.character(where) && length(where) == 1L) return(where)  # already JSON
    as.character(jsonlite::toJSON(where, auto_unbox = TRUE))
}

.rows_to_df <- function(rows, cols) {
    out <- lapply(cols, function(k) {
        if (length(rows) == 0L) return(character(0))
        v <- lapply(rows, function(r) if (is.null(r[[k]])) NA else r[[k]])
        unlist(v, use.names = FALSE)
    })
    names(out) <- cols
    as.data.frame(out, stringsAsFactors = FALSE)
}

#' Search the atlas for cells by their labels and sample metadata
#'
#' Counts and locates cells across every study in the atlas by cell type (Cell Ontology), chromosomal sex, genetic
#' ancestry (HANCESTRO), UMIs and genes per cell, and by the metadata of their sample (tissue, disease, developmental stage,
#' age, organism, assay, chemistry, QC) and study. Ontology terms match their descendants: \code{"T cell"} also finds CD4 and
#' CD8 T cells. The answer comes from the singlet.bio cell index and costs no download; fetch the cells with
#' \code{\link{singlet_fetch_cells}}.
#'
#' @param where A filter: a named list such as \code{list(cell_type = "T cell", sex = "female", tissue = "lung")} (fields are
#'   combined with AND, vector values with OR, a \code{not} element negates), the tree form
#'   \code{list(and = list(...), or = ..., not = ...)}, or the same as a JSON string. Values are ontology ids, labels or
#'   synonyms. The fields and how much of the atlas each covers are listed by \code{singlet_cell_fields()}.
#' @param question Plain English instead of (or ANDed with) \code{where}, read with the server's built-in vocabulary; the
#'   readings are returned so you can check them.
#' @param include_low_qc Also count samples flagged low QC.
#' @param limit,offset Rows of the study and sample tables.
#' @return A list of class \code{singlet_cell_search}: \code{totals} (cells, samples, studies), data frames \code{studies}
#'   and \code{samples} (most cells first), \code{facets} (a breakdown by every other field), \code{unmapped} (for each sample
#'   field filtered on, the served samples with no value for it, which can never match), \code{caveats} and the raw
#'   \code{answer}.
#' @details Cell types below lineage level are previews: per-study accuracy is 0.84 (compartment), 0.75 (lineage), 0.43 (cell
#'   type) and 0.32 (subtype). Genetic ancestry is called only for donors with enough covered sites, and sample metadata such
#'   as disease is recorded for a minority of samples, so a combined filter can match few cells; \code{unmapped} says how
#'   much of the atlas each filter cannot see.
#' @examples
#' \donttest{
#' if (curl::has_internet()) {
#'   hits <- singlet_search_cells(list(cell_type = "T cell", sex = "female", tissue = "blood"), limit = 5)
#'   hits$totals
#'   head(hits$studies)
#' }
#' }
#' @seealso \code{\link{singlet_fetch_cells}}, \code{\link{singlet_cell_fields}}
#' @export
singlet_search_cells <- function(where = NULL, question = NULL, include_low_qc = FALSE, limit = 20L, offset = 0L) {
    if (is.null(where) && is.null(question)) stop("give `where` or `question`", call. = FALSE)
    a <- .singlet_api_json("cells", list(where = .singlet_where_json(where), q = question,
                                         include_low_qc = if (isTRUE(include_low_qc)) "true" else NULL,
                                         limit = as.integer(limit), offset = as.integer(offset)))
    out <- list(
        totals = a$totals,
        studies = .rows_to_df(a$studies, c("gse_id", "cells", "samples")),
        samples = .rows_to_df(a$samples, c("gsm_id", "gse_id", "cells")),
        facets = a$facets,
        unmapped = a$unmapped,
        caveats = unlist(a$caveats),
        basis = a$basis,
        answer = a
    )
    class(out) <- "singlet_cell_search"
    out
}

#' @export
print.singlet_cell_search <- function(x, ...) {
    t <- x$totals
    cat(sprintf("singlet cell search: %s cells in %s samples of %s studies\n", format(t$cells, big.mark = ","),
                format(t$samples, big.mark = ","), format(t$studies, big.mark = ",")))
    if (nrow(x$studies)) {
        cat("Most cells:\n")
        print(utils::head(x$studies, 10L), row.names = FALSE)
    }
    if (length(x$unmapped)) cat("Some filters cannot see every sample: see $unmapped.\n")
    invisible(x)
}

#' The fields cell search understands, and their coverage
#'
#' @return A data frame with one row per field: \code{name}, \code{level} (cell, sample or study), \code{kind}, the share of
#'   cells with a value (\code{cells_share}) and a description.
#' @examples
#' \donttest{
#' if (curl::has_internet()) singlet_cell_fields()[, c("name", "level", "cells_share")]
#' }
#' @rdname singlet_search_cells
#' @export
singlet_cell_fields <- function() {
    a <- .singlet_api_json("cells/fields", what = "cell search fields")
    f <- a$fields
    data.frame(
        name = vapply(f, function(x) x$name, ""),
        level = vapply(f, function(x) x$level %||% NA_character_, ""),
        kind = vapply(f, function(x) x$kind %||% NA_character_, ""),
        cells_share = vapply(f, function(x) as.numeric(x$coverage$cells_share %||% NA_real_), 0),
        description = vapply(f, function(x) x$description %||% NA_character_, ""),
        stringsAsFactors = FALSE
    )
}

`%||%` <- function(a, b) if (is.null(a)) b else a

#' Fetch the cells a search matches
#'
#' Pages through \code{/api/cells/slice} (which cells of which samples match \code{where}) and loads just those cells from
#' their \code{.s1} sample objects, study by study, into one object on the shared gene axis.
#'
#' @inheritParams singlet_search_cells
#' @param max_cells Stop after this many cells (default 10,000). The count is taken in whole samples, so a few more may
#'   be returned.
#' @param as \code{"sce"} (default) or \code{"seurat"}.
#' @param cache_dir Cache directory (see \code{\link{load}}).
#' @param ... Passed to \code{\link{singlet_cells}} (for example \code{layers = TRUE} or \code{cell_type_level}).
#' @return A \code{SingleCellExperiment} (or \code{Seurat}) with one column per cell, \code{colData} as in
#'   \code{\link{singlet_cells}}; or \code{NULL} (invisibly) if nothing matches.
#' @details Cells of one organism can be combined; give \code{organism} in \code{where} when the filter could match several.
#' @examples
#' \donttest{
#' if (curl::has_internet() && requireNamespace("SingleCellExperiment", quietly = TRUE)) {
#'   sce <- singlet_fetch_cells(list(cell_type = "B cell", organism = "Homo sapiens", tissue = "blood"),
#'                              max_cells = 200)
#' }
#' }
#' @seealso \code{\link{singlet_search_cells}}, \code{\link{singlet_gather}}
#' @export
singlet_fetch_cells <- function(where, max_cells = 10000L, include_low_qc = FALSE, as = c("sce", "seurat"),
                                cache_dir = NULL, ...) {
    as <- match.arg(as)
    pages <- singlet_cells_slice(where, max_cells = max_cells, include_low_qc = include_low_qc)
    if (length(pages$cells) == 0L) {
        message("singlet_fetch_cells: no cells match")
        return(invisible(NULL))
    }
    singlet_gather(pages$cells, gse = pages$gse, as = as, cache_dir = cache_dir, ...)
}

#' Which cells of which samples match a filter
#'
#' @inheritParams singlet_fetch_cells
#' @param page_cells Cells per page requested from the server (at most 10,000).
#' @return A list: \code{cells}, a list of \code{list(gsm, columns)} with 0-based matrix columns (the form
#'   \code{\link{singlet_gather}} takes); \code{gse}, a named character vector GSM -> GSE; \code{total}, the cells the filter
#'   matches; and \code{complete}, whether every matching cell was listed.
#' @rdname singlet_fetch_cells
#' @export
singlet_cells_slice <- function(where, max_cells = 10000L, include_low_qc = FALSE, page_cells = 5000L) {
    w <- .singlet_where_json(where)
    cells <- list()
    gse <- character(0)
    n <- 0
    cursor <- NULL
    total <- NA_real_
    repeat {
        p <- .singlet_api_json("cells/slice", list(where = w, cursor = cursor,
                                                   page_cells = as.integer(min(page_cells, max(1, max_cells - n))),
                                                   include_low_qc = if (isTRUE(include_low_qc)) "true" else NULL),
                               what = "cell slice")
        total <- as.numeric(p$progress$cells_total %||% total)
        for (o in p$objects) gse[[o$gsm_id]] <- o$gse_id
        for (cc in p$cells) {
            cols <- as.integer(unlist(cc[[2L]]))
            cells[[length(cells) + 1L]] <- list(gsm = cc[[1L]], columns = cols)
            n <- n + length(cols)
        }
        cursor <- p$next_cursor
        if (is.null(cursor) || n >= max_cells) break
    }
    list(cells = cells, gse = gse, total = total, complete = is.null(cursor))
}

#' Load given cells from any number of samples into one object
#'
#' @param cells A list of \code{list(gsm, columns)} (0-based matrix columns, as \code{\link{singlet_cells_slice}}, the MCP
#'   tools \code{get_cells_page} / \code{find_cells} and Python's \code{atlas.gather()} use), or a named list
#'   \code{GSM = columns}.
#' @param gse Optional named character vector GSM -> GSE; looked up when missing.
#' @inheritParams singlet_fetch_cells
#' @return A \code{SingleCellExperiment} (or \code{Seurat}); \code{colData$matrix_column} is the 1-based column.
#' @examples
#' \donttest{
#' if (curl::has_internet() && requireNamespace("SingleCellExperiment", quietly = TRUE)) {
#'   sce <- singlet_gather(list(list(gsm = "GSM4736413", columns = c(115L, 175L, 260L))))
#' }
#' }
#' @rdname singlet_fetch_cells
#' @export
singlet_gather <- function(cells, gse = NULL, as = c("sce", "seurat"), cache_dir = NULL, ...) {
    as <- match.arg(as)
    if (!is.null(names(cells)) && all(nzchar(names(cells))) && !is.list(cells[[1L]])) {
        cells <- lapply(names(cells), function(g) list(gsm = g, columns = cells[[g]]))
    }
    want <- list()
    for (cc in cells) {
        g <- if (is.null(cc$gsm)) cc[[1L]] else cc$gsm
        cols <- as.integer(if (is.null(cc$columns)) cc[[2L]] else cc$columns)
        want[[g]] <- sort(unique(c(want[[g]], cols)))
    }
    if (length(want) == 0L) stop("no cells given", call. = FALSE)
    cache <- .singlet_cache_dir(cache_dir)
    gse_of <- vapply(names(want), function(g) {
        if (!is.null(gse) && !is.na(gse[g])) unname(gse[g]) else .gse_of(g, cache = cache)
    }, "")
    sces <- lapply(unique(gse_of), function(s) {
        .singlet_cells_by_column(s, want[names(want)[gse_of == s]], cache_dir = cache_dir, ...)
    })
    sce <- if (length(sces) == 1L) sces[[1L]] else .combine_sces(sces)
    if (as == "seurat") return(.sce_to_seurat(sce))
    sce
}

# singlet_cells() for given 0-based columns per sample of one study.
.singlet_cells_by_column <- function(gse, cols_by_gsm, cache_dir = NULL, ...) {
    parts <- lapply(names(cols_by_gsm), function(g) {
        cols <- cols_by_gsm[[g]]
        singlet_cells(gse, where = function(df) (seq_len(nrow(df)) - 1L) %in% cols, samples = g, cache_dir = cache_dir, ...)
    })
    if (length(parts) == 1L) parts[[1L]] else .combine_sces(parts)
}
