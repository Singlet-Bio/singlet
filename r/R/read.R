#' singlet: Read singlet pipeline outputs into R.
#'
#' This package provides R readers for the ``.1pz`` files produced by the
#' singlet single-cell reprocessing pipeline. All decode work happens in
#' a header-only C++ reader shared with the Python wrapper; R only
#' marshals the result into native sparse-matrix classes and single-cell
#' objects.
#'
#' @section Entry points:
#' \describe{
#'   \item{\code{\link{read_1pz}}}{Read a single ``.1pz`` file into a ``dgCMatrix``.}
#'   \item{\code{\link{read_singlet_dir}}}{Read a pipeline output directory into a named list of matrices.}
#'   \item{\code{\link{as_sce}}}{Convert to a ``SingleCellExperiment``.}
#'   \item{\code{\link{as_seurat}}}{Convert to a ``Seurat`` object.}
#' }
#'
#' @keywords internal
#' @aliases singlet-package
"_PACKAGE"


#' Read a single .1pz file into a dgCMatrix.
#'
#' @param path Path to the \code{.1pz} file.
#' @return A \code{\link[Matrix:dgCMatrix-class]{dgCMatrix}} with dimensions
#'   ``features × cells``. The following attributes are attached:
#'   \describe{
#'     \item{\code{user_kv}}{Named character vector of the embedded
#'       pipeline metadata (gsm_id, gse_id, organism, protocol,
#'       singlet_version, pipeline_date, etc.).}
#'     \item{\code{vt_code}}{Writer's value-type hint (1=uint8, 2=uint16, 3=uint32).}
#'   }
#'
#' @examples
#' \dontrun{
#' mat <- read_1pz("quant/scrna/GSE174/GSE174399/GSM5293863/gene_counts.1pz")
#' dim(mat)
#' attr(mat, "user_kv")[["gsm_id"]]
#' }
#'
#' @export
read_1pz <- function(path) {
    raw <- pz_read_1pz_raw(path.expand(as.character(path)))

    shape <- raw$shape
    nrow_ <- shape[1L]
    ncol_ <- shape[2L]

    mat <- Matrix::sparseMatrix(
        i = raw$indices + 1L,          # C++ is 0-indexed, R is 1-indexed
        p = raw$indptr,                 # CSC column pointers are already 0-indexed per dgCMatrix convention
        x = raw$data,
        dims = c(nrow_, ncol_),
        index1 = TRUE,
        repr = "C"
    )
    if (length(raw$rownames) == nrow_) rownames(mat) <- raw$rownames
    if (length(raw$colnames) == ncol_) colnames(mat) <- raw$colnames

    attr(mat, "user_kv") <- raw$user_kv
    attr(mat, "vt_code") <- raw$vt_code
    mat
}


#' Read every .1pz file in a pipeline output directory.
#'
#' @param path Directory containing ``.1pz`` files.
#' @param include Optional character vector of matrix names (without the
#'   ``.1pz`` suffix) to load; default loads all files.
#' @param exclude Optional character vector of matrix names to skip.
#' @return A named list whose elements are \code{\link[Matrix:dgCMatrix-class]{dgCMatrix}}
#'   objects keyed by matrix name. The list has a ``user_kv`` attribute
#'   containing the embedded pipeline metadata from the first-loaded file.
#'
#' @examples
#' \dontrun{
#' dd <- read_singlet_dir("quant/scrna/GSE174/GSE174399/GSM5293863")
#' names(dd)
#' attr(dd, "user_kv")[["gsm_id"]]
#' dd$gene_counts
#' }
#'
#' @export
read_singlet_dir <- function(path, include = NULL, exclude = NULL) {
    path <- path.expand(as.character(path))
    if (!dir.exists(path)) {
        stop(sprintf("not a directory: %s", path))
    }

    files <- list.files(path, pattern = "\\.1pz$", full.names = TRUE)
    if (length(files) == 0L) {
        stop(sprintf("no .1pz files in %s", path))
    }

    out <- list()
    user_kv <- NULL

    for (f in files) {
        name <- sub("\\.1pz$", "", basename(f))
        if (!is.null(include) && !(name %in% include)) next
        if (!is.null(exclude) && (name %in% exclude)) next

        mat <- read_1pz(f)
        out[[name]] <- mat
        if (is.null(user_kv)) user_kv <- attr(mat, "user_kv")
    }

    attr(out, "user_kv") <- user_kv
    attr(out, "path") <- path
    class(out) <- c("singlet_dir", class(out))
    out
}


#' Bulk-read the same matrix from many singlet pipeline output directories.
#'
#' Cohort-level convenience reader. Walks a list of GSM sample directories
#' and loads the same matrix from each. The most common use case: you have
#' a GSE directory with N GSM subdirectories and want the spliced matrix
#' from each to merge into a single sparse matrix for cross-sample analysis.
#'
#' @param paths Character vector of pipeline output directories.
#' @param matrix_name Which matrix to load (without ``.1pz`` suffix). Default ``"spliced"``.
#' @param show_progress When TRUE, print one-line-per-sample progress to stderr.
#' @return A named list of \code{Matrix::dgCMatrix} objects keyed by gsm_id.
#'   Directories that lack the requested matrix are silently dropped.
#'
#' @examples
#' \dontrun{
#' # Load all spliced matrices from a GSE
#' gse_dir <- "quant/scrna/GSE174/GSE174399"
#' sample_dirs <- list.dirs(gse_dir, recursive = FALSE)
#' cohort <- read_cohort(sample_dirs, matrix_name = "spliced")
#'
#' # Concatenate column-wise into one big matrix
#' library(Matrix)
#' big <- do.call(cbind, cohort)
#' dim(big)
#' }
#'
#' @export
read_cohort <- function(paths, matrix_name = "spliced", show_progress = FALSE) {
    out <- list()
    for (i in seq_along(paths)) {
        pdir <- path.expand(as.character(paths[i]))
        if (show_progress) {
            message(sprintf("[%d/%d] %s", i, length(paths), basename(pdir)))
        }
        target <- file.path(pdir, paste0(matrix_name, ".1pz"))
        if (!file.exists(target)) {
            if (show_progress) {
                message(sprintf("  skip — %s.1pz not present", matrix_name))
            }
            next
        }
        mat <- read_1pz(target)
        gsm <- attr(mat, "user_kv")[["gsm_id"]]
        if (is.null(gsm) || nchar(gsm) == 0L) {
            gsm <- basename(pdir)
        }
        out[[gsm]] <- mat
    }
    out
}


#' @export
print.singlet_dir <- function(x, ...) {
    gsm <- attr(x, "user_kv")
    gsm_id <- if (!is.null(gsm) && "gsm_id" %in% names(gsm)) gsm[["gsm_id"]] else "?"
    cat(sprintf("singlet pipeline directory (%d matrices, gsm=%s)\n",
                length(x), gsm_id))
    for (nm in names(x)) {
        d <- dim(x[[nm]])
        nnz <- length(x[[nm]]@x)
        cat(sprintf("  %-20s %d x %d, nnz=%d\n", nm, d[1], d[2], nnz))
    }
    invisible(x)
}
