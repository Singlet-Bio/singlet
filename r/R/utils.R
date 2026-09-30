# SPDX-License-Identifier: MIT
# Internal helpers shared across the package: optional-dependency loading,
# tolerant JSON field access, and cell-call parsing.


# ---------------------------------------------------------------------------
# Internal: load Suggested packages and fail with the real reason.
#
# requireNamespace(quietly = TRUE) only says FALSE. When a package is
# installed but cannot be loaded (a missing dependency of its own, a version
# mismatch after an R upgrade, a half-finished install) that hides the one
# thing the user needs to see, so we call loadNamespace() and pass its error
# message through alongside the install line.
# ---------------------------------------------------------------------------
.require_namespaces <- function(pkgs, caller, install) {
    for (pkg in pkgs) {
        problem <- tryCatch({
            loadNamespace(pkg)
            NULL
        }, error = function(e) conditionMessage(e))
        if (!is.null(problem)) {
            stop(sprintf(
                "%s needs the '%s' package, which could not be loaded:\n  %s\nInstall it with:\n  %s",
                caller, pkg, problem, install
            ), call. = FALSE)
        }
    }
    invisible(TRUE)
}

.bioc_install_hint <- paste0(
    "install.packages(\"BiocManager\"); ",
    "BiocManager::install(c(\"SingleCellExperiment\", ",
    "\"SummarizedExperiment\", \"S4Vectors\"))"
)

# Everything needed to build a SingleCellExperiment.
.require_sce <- function(caller) {
    .require_namespaces(
        c("SingleCellExperiment", "SummarizedExperiment", "S4Vectors"),
        caller, .bioc_install_hint
    )
}

.require_seurat <- function(caller) {
    .require_namespaces("Seurat", caller, "install.packages(\"Seurat\")")
}


# ---------------------------------------------------------------------------
# Internal: walk a parsed JSON document (simplifyVector = FALSE) along
# `path`, returning NULL as soon as a level is missing or not an object.
# Uses [[ ]] rather than $ so a missing key never partially matches another.
# ---------------------------------------------------------------------------
.json_field <- function(x, path) {
    for (key in path) {
        if (!is.list(x) || is.null(x[[key]])) {
            return(NULL)
        }
        x <- x[[key]]
    }
    x
}


# ---------------------------------------------------------------------------
# Internal: cell calls.
#
# cell_calls.tsv has spelled its barcode column several ways over the
# pipeline's history, and is_cell has been written as TRUE/FALSE,
# True/False and 1/0. These mirror the Python reader
# (`singlet.bundle._get_called_barcodes`).
# ---------------------------------------------------------------------------
.barcode_columns <- c("barcode", "cb", "cell_barcode", "CB")

# Coerce an is_cell column to logical; anything unrecognised is FALSE.
.as_flag <- function(x) {
    out <- if (is.logical(x)) {
        x
    } else if (is.numeric(x)) {
        x != 0
    } else {
        tolower(trimws(as.character(x))) %in% c("true", "t", "1", "yes", "y")
    }
    out[is.na(out)] <- FALSE
    out
}

# Barcodes a cell_calls table calls as cells, or NULL when the table cannot
# say (no usable barcode column). A table with a barcode column but no rows,
# or with every is_cell FALSE, yields character(0): the pipeline looked and
# found no cells.
.called_barcodes_from_table <- function(cc) {
    if (is.null(cc) || !is.data.frame(cc) || ncol(cc) == 0L) {
        return(NULL)
    }
    col <- intersect(.barcode_columns, colnames(cc))
    if (length(col) == 0L) {
        # Same fallback as the Python reader: the first column, unless that
        # is the flag itself.
        first <- colnames(cc)[[1L]]
        if (identical(first, "is_cell")) {
            return(NULL)
        }
        col <- first
    }
    bc <- as.character(cc[[col[[1L]]]])
    if ("is_cell" %in% colnames(cc)) {
        bc <- bc[.as_flag(cc[["is_cell"]])]
    }
    bc
}

# Read cell_calls.tsv from disk. Every column is read as character so
# barcodes that look numeric (plate wells, "001") are not mangled. Returns
# NULL when the file is absent or unreadable.
.read_cell_calls <- function(path) {
    if (!file.exists(path)) {
        return(NULL)
    }
    tryCatch(
        utils::read.table(path, sep = "\t", header = TRUE,
                          colClasses = "character", quote = "",
                          comment.char = "", stringsAsFactors = FALSE,
                          check.names = FALSE),
        error = function(e) NULL
    )
}
