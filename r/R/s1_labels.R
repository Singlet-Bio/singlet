# SPDX-License-Identifier: MIT
# Per-cell columns of a .s1 sample as colData: the Research Alpha labels, the cell calls and the QC columns of the cells table
# (the same names and rules as python/singlet/s1/labels.py; documented in ?singlet_labels).

#' Per-cell labels of .s1 studies
#'
#' \code{\link{load}} / \code{singlet_load()} and \code{\link{singlet_cells}} put the per-cell columns of every sample's cells
#' table into \code{colData}, in front of the GEO sample metadata:
#' \describe{
#'   \item{\code{cell_type}, \code{cell_type_ontology_term_id}}{Cell Ontology name and id at the default level: the deepest level
#'     \code{k <= cell_type_level} (default 2) that the labeller's gating reached; with \code{cell_type_min_score > 0} (default 0)
#'     the path score \code{p1 * ... * pk} must also reach that value. \code{NA} when the cell has no call.}
#'   \item{\code{cell_type_level}, \code{cell_type_score}}{that level (1 compartment, 2 lineage, 3 cell type, 4+ subtype; 0 none) and
#'     its path score.}
#'   \item{\code{cell_type_deepest}, \code{cell_type_deepest_ontology_term_id}, \code{cell_type_deepest_level},
#'     \code{cell_type_deepest_score}}{the labeller's deepest call (levels 3 and deeper are previews).}
#'   \item{\code{cell_type_path}, \code{cell_type_p1} .. \code{cell_type_p6}}{its full CL path (ids joined by \code{>}, compartment
#'     first) and the gating confidence of each level.}
#'   \item{\code{sex_chrom}, \code{sex_score}}{chromosomal sex (\code{female}, \code{male}, \code{ambiguous}) and its posterior.}
#'   \item{\code{ancestry_hancestro}, \code{ancestry_prob}}{continental genetic ancestry of the cell's donor (HANCESTRO id) and its
#'     proportion; human donors with per-donor coverage tables only.}
#'   \item{\code{called}, \code{called_ed}, \code{called_ed_fdr}}{the default cell call, and the EmptyDrops call (additional).}
#'   \item{\code{n_umi_*}, \code{n_feat_*}, \code{cyc_*}, \code{qc_*}, \code{don_*}, \code{mts_*}}{counts summaries, cell cycle, QC,
#'     genotype-free donor assignment and mtDNA summary.}
#' }
#' GEO sample metadata follows: \code{organism}, \code{protocol}, \code{protocol_name}, \code{sample_source},
#' \code{sample_characteristics}, \code{qc_flag}, \code{reference_build} and the curated fields \code{geo_tissue},
#' \code{geo_cell_type}, \code{geo_disease}, \code{geo_sex}, \code{geo_age} (sample-level; before 1.1.0 these were named
#' \code{tissue}, \code{cell_type}, ... and collided with the per-cell labels).
#'
#' \strong{Accuracy} (docs/cell_annotation.md): cell type per-study accuracy against CELLxGENE author labels is 0.84 at level 1,
#' 0.75 at level 2, 0.43 at level 3 and 0.32 at level 4, so levels 3+ are previews; the score ranks calls and is optimistic above
#' 0.9. Sex agrees with GEO metadata for 153 of 156 called samples (98.1\%). Ancestry is continental (HANCESTRO), for human donors
#' with coverage only (about 1\% of cells).
#'
#' \strong{Hidden by default:} \code{qc_doublet_score} is unvalidated and is left out unless \code{include_unvalidated = TRUE} or
#' it is named in \code{columns}; \code{where} conditions can still use it.
#'
#' @name singlet_labels
#' @seealso \code{\link{load}}, \code{\link{singlet_cells}}
NULL

.s1_label_max_levels <- 6L
.s1_label_lead <- c("cell_type", "cell_type_ontology_term_id", "cell_type_level", "cell_type_score", "cell_type_deepest",
                    "cell_type_deepest_ontology_term_id", "cell_type_deepest_level", "cell_type_deepest_score", "cell_type_path",
                    paste0("cell_type_p", 1:6), "sex_chrom", "sex_score", "ancestry_hancestro", "ancestry_prob", "called", "called_ed",
                    "called_ed_fdr")
.s1_label_renames <- c(cell_type_cl = "cell_type_deepest_ontology_term_id", cell_type_label = "cell_type_deepest",
                       cell_type_score = "cell_type_deepest_score")
.s1_label_unvalidated <- c("qc_doublet_score")
.s1_label_internal <- c("barcode", "cc_pos", "cc_listed", "cc_pvalue")

.s1_cl_names <- local({
    cache <- NULL
    function() {
        if (is.null(cache)) {
            p <- system.file("extdata", "cl_names.tsv", package = "singlet")
            tab <- if (nzchar(p)) utils::read.delim(p, comment.char = "#", quote = "", stringsAsFactors = FALSE,
                                                    colClasses = "character") else data.frame(id = character(0), name = character(0))
            cache <<- structure(tab$name, names = tab$id)
        }
        cache
    }
})

# Labels of the categorical cells-table columns (stored as codes), from the recipes of a sample's derived manifest members.
.s1_label_categories <- function(members) {
    out <- list()
    for (m in members) {
        cs <- m$params$c
        if (!is.list(cs)) next
        for (c in cs) {
            if (!is.list(c) || !identical(c$s, "c") || is.null(c$c) || !identical(c$f$k, "cat") || is.null(c$f$labels)) next
            if (is.null(out[[c$c]])) out[[c$c]] <- as.character(unlist(c$f$labels))
        }
    }
    out
}

.s1_na_strings <- function(v) {
    v <- as.character(v)
    v[!is.na(v) & !nzchar(v)] <- NA_character_
    v
}

.s1_cell_type_columns <- function(cols, n, level = 2L, min_score = 0) {
    if (is.null(cols$cell_type_path)) return(list())
    L <- .s1_label_max_levels
    path <- .s1_na_strings(cols$cell_type_path)
    deep_id <- if (is.null(cols$cell_type_cl)) rep(NA_character_, n) else .s1_na_strings(cols$cell_type_cl)
    deep_name <- if (is.null(cols$cell_type_label)) rep(NA_character_, n) else .s1_na_strings(cols$cell_type_label)
    uniq <- unique(path[!is.na(path)])
    ids <- matrix(NA_character_, length(uniq) + 1L, L)
    depth_u <- integer(length(uniq) + 1L)
    for (u in seq_along(uniq)) {
        parts <- utils::head(strsplit(uniq[[u]], ">", fixed = TRUE)[[1L]], L)
        ids[u, seq_along(parts)] <- parts
        depth_u[[u]] <- length(parts)
    }
    code <- match(path, uniq)
    code[is.na(code)] <- length(uniq) + 1L
    depth <- depth_u[code]
    P <- matrix(NA_real_, n, L)
    for (k in seq_len(L)) {
        v <- cols[[paste0("cell_type_p", k)]]
        if (!is.null(v)) P[, k] <- as.numeric(v)
    }
    P[is.nan(P)] <- NA_real_
    S <- P
    S[is.na(S)] <- 0
    if (L > 1L) for (k in 2:L) S[, k] <- S[, k - 1L] * S[, k]
    lv <- max(0L, min(as.integer(level), L))
    ok <- matrix(FALSE, n, L)
    for (k in seq_len(L)) ok[, k] <- k <= lv & k <= depth & !is.na(P[, k]) & S[, k] >= min_score
    if (L > 1L) for (k in 2:L) ok[, k] <- ok[, k] & ok[, k - 1L]
    chosen <- as.integer(rowSums(ok))
    rows <- which(chosen > 0L)
    ct_id <- rep(NA_character_, n)
    ct_score <- rep(NA_real_, n)
    ct_id[rows] <- ids[cbind(code[rows], chosen[rows])]
    ct_score[rows] <- S[cbind(rows, chosen[rows])]
    names_cl <- .s1_cl_names()
    local_names <- structure(deep_name, names = deep_id)
    nm <- unname(names_cl[ct_id])
    miss <- is.na(nm) & !is.na(ct_id)
    nm[miss] <- unname(local_names[ct_id[miss]])
    miss <- is.na(nm) & !is.na(ct_id)
    nm[miss] <- ct_id[miss]
    deep_score <- if (is.null(cols$cell_type_score)) rep(NA_real_, n) else as.numeric(cols$cell_type_score)
    deep_score[is.nan(deep_score)] <- NA_real_
    list(cell_type = nm, cell_type_ontology_term_id = ct_id, cell_type_level = chosen, cell_type_score = ct_score,
         cell_type_deepest = deep_name, cell_type_deepest_ontology_term_id = deep_id,
         cell_type_deepest_level = ifelse(is.na(deep_id), 0L, depth), cell_type_deepest_score = deep_score,
         cell_type_path = path)
}

# The colData columns (a named list of length-n vectors) of one sample's selected cells. `cols` are the stored cells-table
# columns restricted to those cells; `cats` the labels of the categorical ones. `columns` keeps only those names (stored names
# are accepted too); otherwise every column except the internal ones and, without `include_unvalidated`, the unvalidated ones.
.s1_label_columns <- function(cols, n, cats = list(), cell_type_level = 2L, cell_type_min_score = 0,
                              include_unvalidated = FALSE, columns = NULL) {
    out <- .s1_cell_type_columns(cols, n, cell_type_level, cell_type_min_score)
    for (name in names(cols)) {
        if (name %in% c(.s1_label_internal, names(.s1_label_renames), "cell_type_path")) next
        v <- cols[[name]]
        if (is.list(v)) next  # an unresolved whitelist-coded column
        if (!is.null(cats[[name]]) && is.numeric(v)) {
            lab <- cats[[name]]
            i <- as.integer(v) + 1L
            i[!is.na(i) & (i < 1L | i > length(lab))] <- NA_integer_
            v <- lab[i]
        } else if (is.character(v)) {
            v <- .s1_na_strings(v)
        } else if (is.numeric(v)) {
            v[is.nan(v)] <- NA_real_
        }
        out[[name]] <- v
    }
    if (!is.null(columns)) {
        want <- ifelse(columns %in% names(.s1_label_renames), .s1_label_renames[columns], columns)
        out <- out[intersect(names(out), want)]
    } else if (!isTRUE(include_unvalidated)) {
        out <- out[setdiff(names(out), .s1_label_unvalidated)]
    }
    out[c(intersect(.s1_label_lead, names(out)), setdiff(names(out), .s1_label_lead))]
}

# Row-bind per-sample column lists into one data.frame; a column a sample lacks is NA for its cells.
.s1_label_rbind <- function(parts) {
    nms <- unique(unlist(lapply(parts, names), use.names = FALSE))
    lens <- vapply(parts, function(p) if (length(p)) length(p[[1L]]) else 0L, integer(1))
    cols <- lapply(nms, function(nm) {
        proto <- NULL
        for (p in parts) if (!is.null(p[[nm]])) { proto <- p[[nm]]; break }
        na <- if (is.logical(proto)) NA else if (is.character(proto)) NA_character_ else if (is.integer(proto)) NA_integer_ else NA_real_
        unlist(lapply(seq_along(parts), function(k) if (is.null(parts[[k]][[nm]])) rep(na, lens[[k]]) else parts[[k]][[nm]]),
               use.names = FALSE)
    })
    names(cols) <- nms
    as.data.frame(cols, stringsAsFactors = FALSE, check.names = FALSE)
}
