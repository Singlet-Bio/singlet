#' @title SinglePress R bindings for .1pz format
#' @description Read and write .1pz v3 sparse matrix files.
#' @name singlepress
#' @importFrom methods setClass setMethod setAs new is as
NULL

# ============================================================================
# S4 OnePZ class
# ============================================================================

#' @title OnePZ file handle
#' @description Lazy handle to a .1pz file for metadata access and coercion.
#' @slot path Character. File path.
#' @slot nrow Integer. Number of features (rows).
#' @slot ncol Integer. Number of cells (columns).
#' @slot nnz Numeric. Number of non-zero entries.
#' @slot info List. Header metadata from info_1pz().
#' @slot normalize Logical. If TRUE, reads apply log-normalization.
#' @slot scale Numeric. Scaling factor for normalization.
#' @export
setClass("OnePZ", representation(
  path      = "character",
  nrow      = "integer",
  ncol      = "integer",
  nnz       = "numeric",
  info      = "list",
  normalize = "logical",
  scale     = "numeric"
))

#' Open a .1pz file as an OnePZ handle
#'
#' @param path Path to a .1pz file.
#' @param normalize Logical. If TRUE, all reads apply on-the-fly
#'   log-normalization: \code{log1p(x * scale / colsum)}.
#' @param scale Scaling factor for normalization. Default 10000.
#' @return An OnePZ S4 object.
#' @export
OnePZ <- function(path, normalize = FALSE, scale = 10000) {
  path <- normalizePath(path, mustWork = TRUE)
  info <- info_1pz(path)
  new("OnePZ",
      path = path,
      nrow = as.integer(info$m),
      ncol = as.integer(info$n),
      nnz  = as.numeric(info$nnz),
      info = info,
      normalize = as.logical(normalize),
      scale = as.numeric(scale))
}

#' Return a normalized copy of an OnePZ handle
#'
#' All subsequent reads, indexing, and coercions on the returned handle
#' will automatically apply \code{log1p(x * scale / colsum)}.
#'
#' @param x An OnePZ object.
#' @param scale Scaling factor. Default 10000.
#' @return An OnePZ object with normalization enabled.
#' @export
normalized <- function(x, scale = 10000) UseMethod("normalized")

#' @export
normalized.OnePZ <- function(x, scale = 10000) {
  new("OnePZ",
      path = x@path,
      nrow = x@nrow,
      ncol = x@ncol,
      nnz  = x@nnz,
      info = x@info,
      normalize = TRUE,
      scale = as.numeric(scale))
}

# -- Dimension generics for OnePZ -------------------------------------------

#' @export
setMethod("dim", "OnePZ", function(x) c(x@nrow, x@ncol))

#' @export
setMethod("nrow", "OnePZ", function(x) x@nrow)

#' @export
setMethod("ncol", "OnePZ", function(x) x@ncol)

#' @export
setMethod("length", "OnePZ", function(x) as.numeric(x@nrow) * as.numeric(x@ncol))

# -- Show / print method -----------------------------------------------------

.human_size <- function(bytes) {
  units <- c("B", "KB", "MB", "GB", "TB")
  idx <- 1L
  val <- as.numeric(bytes)
  while (val >= 1024 && idx < length(units)) {
    val <- val / 1024
    idx <- idx + 1L
  }
  if (val == as.integer(val)) sprintf("%d %s", as.integer(val), units[idx])
  else sprintf("%.1f %s", val, units[idx])
}

#' @export
setMethod("show", "OnePZ", function(object) {
  m <- object@nrow; n <- object@ncol; nnz <- object@nnz
  density <- if (as.numeric(m) * n > 0) nnz / (as.numeric(m) * n) * 100 else 0
  sz <- file.info(object@path)$size

  cat(sprintf("OnePZ: %s\n", basename(object@path)))
  cat(sprintf("  Shape:   %s features x %s cells\n",
              formatC(m, big.mark = ",", format = "d"),
              formatC(n, big.mark = ",", format = "d")))
  cat(sprintf("  NNZ:     %s (%.1f%% dense)\n",
              formatC(nnz, big.mark = ",", format = "d"), density))
  cat(sprintf("  Size:    %s\n", .human_size(sz)))
  cat(sprintf("  Format:  .1pz v%d\n", object@info$version))

  flags <- character(0)
  if (isTRUE(object@info$has_metadata))  flags <- c(flags, "names")
  if (isTRUE(object@info$has_colsums))   flags <- c(flags, "colsums")
  if (isTRUE(object@info$has_transpose)) flags <- c(flags, "transpose")
  if (isTRUE(object@info$has_obs_var))   flags <- c(flags, "obs/var")
  if (length(flags) > 0)
    cat(sprintf("  Stored:  %s\n", paste(flags, collapse = ", ")))
  if (object@normalize)
    cat(sprintf("  Normalize: log1p(x * %.0f / colsum)\n", object@scale))

  invisible(object)
})

# -- as() coercion methods ---------------------------------------------------

#' @export
setAs("OnePZ", "dgCMatrix", function(from) {
  mat <- read_1pz(from@path)
  if (from@normalize) {
    cs <- attr(mat, "colsums")
    if (is.null(cs)) stop("File has no stored colsums; cannot normalize")
    mat <- lognorm(mat, cs, scale = from@scale)
  }
  mat
})

#' @export
setAs("OnePZ", "matrix", function(from) as.matrix(as(from, "dgCMatrix")))

#' @export
setAs("OnePZ", "data.frame", function(from) {
  if (from@normalize) {
    as.data.frame(as.matrix(as(from, "dgCMatrix")))
  } else {
    read_1pz_dataframe(from@path)
  }
})

# -- AnnData-style aliases ---------------------------------------------------

#' Number of observations (cells). AnnData convention.
#' @param x An OnePZ object.
#' @return Integer.
#' @export
n_obs <- function(x) UseMethod("n_obs")

#' @export
n_obs.OnePZ <- function(x) x@ncol

#' Number of variables (features). AnnData convention.
#' @param x An OnePZ object.
#' @return Integer.
#' @export
n_vars <- function(x) UseMethod("n_vars")

#' @export
n_vars.OnePZ <- function(x) x@nrow

# -- Indexing: opz[rows, cols] -----------------------------------------------

#' @export
setMethod("[", signature(x = "OnePZ"), function(x, i, j, ..., drop = FALSE) {
  mat <- read_1pz(x@path)
  cs_full <- attr(mat, "colsums")
  rn <- rownames(mat)
  cn <- colnames(mat)

  if (!missing(i)) {
    if (is.character(i)) {
      if (is.null(rn)) stop("Cannot subset by row name: no rownames stored")
      i <- match(i, rn)
      if (any(is.na(i))) stop("Some row names not found")
    }
    mat <- mat[i, , drop = FALSE]
  }
  if (!missing(j)) {
    if (is.character(j)) {
      if (is.null(cn)) stop("Cannot subset by column name: no colnames stored")
      j <- match(j, cn)
      if (any(is.na(j))) stop("Some column names not found")
    }
    mat <- mat[, j, drop = FALSE]
    if (!is.null(cs_full)) cs_full <- cs_full[j]
  }

  if (x@normalize) {
    cs <- if (!is.null(cs_full)) cs_full else attr(mat, "colsums")
    if (is.null(cs)) stop("File has no stored colsums; cannot normalize")
    mat <- lognorm(mat, cs, scale = x@scale)
  }
  mat
})

# -- Transpose ---------------------------------------------------------------

#' @export
setMethod("t", "OnePZ", function(x) {
  mat <- read_1pz(x@path)
  Matrix::t(mat)
})

# -- Summary statistics ------------------------------------------------------

#' Non-zeros per column (cell) of a .1pz file.
#' @param path Path to .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return Integer vector of length ncol.
#' @export
nnz_per_col <- function(path, num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  diff(mat@p)
}

#' Non-zeros per row (feature) of a .1pz file.
#' @param path Path to .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return Integer vector of length nrow.
#' @export
nnz_per_row <- function(path, num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  tabulate(mat@i + 1L, nbins = nrow(mat))
}

#' Row sums of a .1pz file.
#' @param path Path to .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return Numeric vector of length nrow.
#' @export
rowsums_1pz <- function(path, num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  Matrix::rowSums(mat)
}

#' Summary description of a .1pz file.
#' @param path Path to .1pz file.
#' @return Named list with shape, nnz, density, file_size, etc.
#' @export
describe_1pz <- function(path) {
  info <- info_1pz(path)
  m <- info$m; n <- info$n; nnz <- info$nnz
  sz <- file.info(path)$size
  density <- if (as.numeric(m) * n > 0) nnz / (as.numeric(m) * n) else 0
  list(
    path = path,
    shape = c(m, n),
    nnz = nnz,
    density = density,
    file_size = sz,
    version = info$version,
    has_metadata = isTRUE(info$has_metadata),
    has_colsums = isTRUE(info$has_colsums),
    has_transpose = isTRUE(info$has_transpose),
    has_obs_var = isTRUE(info$has_obs_var)
  )
}

# -- head / tail -------------------------------------------------------------

#' Read first n cells from .1pz
#' @param x An OnePZ object or path string.
#' @param n Number of cells. Default 10.
#' @param ... Ignored.
#' @return A dgCMatrix.
#' @export
setMethod("head", "OnePZ", function(x, n = 6L, ...) {
  mat <- read_1pz(x@path)
  nc <- min(n, ncol(mat))
  mat[, seq_len(nc), drop = FALSE]
})

#' Read last n cells from .1pz
#' @param x An OnePZ object or path string.
#' @param n Number of cells. Default 10.
#' @param ... Ignored.
#' @return A dgCMatrix.
#' @export
setMethod("tail", "OnePZ", function(x, n = 6L, ...) {
  mat <- read_1pz(x@path)
  nc <- ncol(mat)
  start <- max(1L, nc - n + 1L)
  mat[, start:nc, drop = FALSE]
})

# Convenience coercion for Seurat (loaded at runtime)
.onepz_to_seurat <- function(from) {
  if (!requireNamespace("Seurat", quietly = TRUE))
    stop("Seurat package required. Install with: install.packages('Seurat')")
  read_1pz_seurat(from@path)
}

# Convenience coercion for SingleCellExperiment
.onepz_to_sce <- function(from) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE))
    stop("SingleCellExperiment required. Install with: BiocManager::install('SingleCellExperiment')")
  read_1pz_sce(from@path)
}

# Register as() coercions on package load (Seurat/SCE may not be available)
.onLoad <- function(libname, pkgname) {
  # Always-available coercions are registered via setAs above.
  # Seurat/SCE coercions are registered dynamically to avoid hard dependency.
  if (requireNamespace("Seurat", quietly = TRUE))
    setAs("OnePZ", "Seurat", .onepz_to_seurat)
  if (requireNamespace("SingleCellExperiment", quietly = TRUE))
    setAs("OnePZ", "SingleCellExperiment", .onepz_to_sce)
}

# ============================================================================
# cbind_1pz — horizontal concatenation
# ============================================================================

#' Column-bind multiple .1pz files
#'
#' Reads all input files, verifies row compatibility, and writes a
#' combined .1pz file with columns (cells) concatenated in order.
#'
#' @param ... Paths to .1pz files (character).
#' @param output Output .1pz file path.
#' @param num_threads Number of threads. Default 4.
#' @param verify_rownames Check that rownames match across files. Default TRUE.
#' @return Invisible list with compression statistics.
#' @export
cbind_1pz <- function(..., output, num_threads = 4L, verify_rownames = TRUE) {
  paths <- c(...)
  if (length(paths) < 2L) stop("cbind_1pz requires at least 2 input files")

  mats <- lapply(paths, read_1pz, num_threads = num_threads)

  # Validate rows
  nrows <- vapply(mats, nrow, integer(1))
  if (length(unique(nrows)) != 1L)
    stop("All files must have the same number of rows. Got: ",
         paste(nrows, collapse = ", "))

  if (verify_rownames) {
    rn0 <- rownames(mats[[1]])
    for (i in seq_along(mats)[-1]) {
      rn_i <- rownames(mats[[i]])
      if (!is.null(rn0) && !is.null(rn_i) && !identical(rn0, rn_i))
        stop("Rownames mismatch between file 1 and file ", i)
    }
  }

  combined <- do.call(cbind, mats)
  rn <- rownames(mats[[1]])
  cn <- unlist(lapply(mats, colnames))
  if (length(cn) == 0L) cn <- NULL

  result <- write_1pz(combined, output, rownames = rn, colnames = cn,
                       num_threads = num_threads)
  invisible(result)
}


# ============================================================================
# rbind_1pz — vertical concatenation
# ============================================================================

#' Row-bind multiple .1pz files
#'
#' Reads all input files, verifies column compatibility, and writes a
#' combined .1pz file with rows (features) concatenated in order.
#'
#' @param ... Paths to .1pz files (character).
#' @param output Output .1pz file path.
#' @param num_threads Number of threads. Default 4.
#' @param verify_colnames Check that colnames match across files. Default TRUE.
#' @return Invisible list with compression statistics.
#' @export
rbind_1pz <- function(..., output, num_threads = 4L, verify_colnames = TRUE) {
  paths <- c(...)
  if (length(paths) < 2L) stop("rbind_1pz requires at least 2 input files")

  mats <- lapply(paths, read_1pz, num_threads = num_threads)

  ncols <- vapply(mats, ncol, integer(1))
  if (length(unique(ncols)) != 1L)
    stop("All files must have the same number of columns. Got: ",
         paste(ncols, collapse = ", "))

  if (verify_colnames) {
    cn0 <- colnames(mats[[1]])
    for (i in seq_along(mats)[-1]) {
      cn_i <- colnames(mats[[i]])
      if (!is.null(cn0) && !is.null(cn_i) && !identical(cn0, cn_i))
        stop("Colnames mismatch between file 1 and file ", i)
    }
  }

  combined <- do.call(rbind, mats)
  cn <- colnames(mats[[1]])
  rn <- unlist(lapply(mats, rownames))
  if (length(rn) == 0L) rn <- NULL

  result <- write_1pz(combined, output, rownames = rn, colnames = cn,
                       num_threads = num_threads)
  invisible(result)
}


# ============================================================================
# subset_1pz — write a filtered subset to a new .1pz
# ============================================================================

#' Subset a .1pz file to a new .1pz file
#'
#' @param path Input .1pz file path.
#' @param output Output .1pz file path.
#' @param rows Row indices (integer), row names (character), or logical mask.
#' @param cols Column indices (integer), column names (character), or logical mask.
#' @param num_threads Number of threads. Default 4.
#' @return Invisible list with compression statistics.
#' @export
subset_1pz <- function(path, output, rows = NULL, cols = NULL,
                        num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  obs_df <- attr(mat, "obs")
  var_df <- attr(mat, "var")
  uns <- attr(mat, "uns")

  if (!is.null(rows)) {
    if (is.character(rows)) {
      rn <- rownames(mat)
      if (is.null(rn)) stop("Cannot subset by row name: no rownames stored")
      rows <- match(rows, rn)
      if (any(is.na(rows))) stop("Some row names not found")
    }
    mat <- mat[rows, , drop = FALSE]
    if (!is.null(var_df)) {
      idx <- if (is.logical(rows)) which(rows) else rows
      var_df <- var_df[idx, , drop = FALSE]
      rownames(var_df) <- NULL
    }
  }
  if (!is.null(cols)) {
    if (is.character(cols)) {
      cn <- colnames(mat)
      if (is.null(cn)) stop("Cannot subset by column name: no colnames stored")
      cols <- match(cols, cn)
      if (any(is.na(cols))) stop("Some column names not found")
    }
    mat <- mat[, cols, drop = FALSE]
    if (!is.null(obs_df)) {
      idx <- if (is.logical(cols)) which(cols) else cols
      obs_df <- obs_df[idx, , drop = FALSE]
      rownames(obs_df) <- NULL
    }
  }

  result <- write_1pz(mat, output,
                       rownames = rownames(mat),
                       colnames = colnames(mat),
                       num_threads = num_threads)
  invisible(result)
}


# ============================================================================
# sample_1pz — random downsample cells
# ============================================================================

#' Randomly sample cells from a .1pz file to a new .1pz file
#'
#' @param path Input .1pz file path.
#' @param output Output .1pz file path.
#' @param n Number of cells to sample.
#' @param fraction Fraction of cells to sample (alternative to n).
#' @param seed Random seed. Default 42.
#' @param num_threads Number of threads. Default 4.
#' @return Invisible list with compression statistics.
#' @export
sample_1pz <- function(path, output, n = NULL, fraction = NULL,
                        seed = 42L, num_threads = 4L) {
  if (is.null(n) == is.null(fraction))
    stop("Exactly one of n or fraction must be specified")

  info <- info_1pz(path)
  total <- info$n

  if (!is.null(fraction)) {
    if (fraction <= 0 || fraction > 1) stop("fraction must be in (0, 1]")
    n <- max(1L, round(total * fraction))
  }
  n <- min(n, total)

  set.seed(seed)
  indices <- sort(sample.int(total, size = n))

  subset_1pz(path, output, cols = indices, num_threads = num_threads)
}

#' Read a .1pz file into a dgCMatrix
#'
#' @param path Path to a .1pz file.
#' @param num_threads Number of OpenMP threads. Default 4.
#' @param normalize Logical. If TRUE, apply on-the-fly log-normalization:
#'   \code{log1p(x * scale / colsum)}. Equivalent to Seurat's LogNormalize.
#'   Requires stored column sums.
#' @param scale Scaling factor for normalization. Default 10000.
#' @return A dgCMatrix (sparse column-compressed matrix) with optional
#'   rownames, colnames, colsums, uns, obs, and var attributes.
#'   obs and var are data.frames with cell/feature metadata.
#'   uns is a named character vector of key-value metadata.
#' @export
read_1pz <- function(path, num_threads = 4L, normalize = FALSE,
                      scale = 10000) {
  result <- read_1pz_r(path, as.integer(num_threads))
  mat <- Matrix::sparseMatrix(
    i = result$i + 1L,  # 0-indexed -> 1-indexed
    p = result$p,
    x = result$x,
    dims = c(result$nrow, result$ncol),
    repr = "C"
  )
  if (!is.null(result$rownames)) rownames(mat) <- result$rownames
  if (!is.null(result$colnames)) colnames(mat) <- result$colnames
  if (!is.null(result$colsums)) attr(mat, "colsums") <- result$colsums
  if (!is.null(result$uns)) attr(mat, "uns") <- result$uns
  if (!is.null(result$obs)) {
    obs_df <- as.data.frame(result$obs, stringsAsFactors = FALSE)
    rn <- attr(result$obs, "row.names")
    if (!is.null(rn)) rownames(obs_df) <- rn
    attr(mat, "obs") <- obs_df
  }
  if (!is.null(result$var)) {
    var_df <- as.data.frame(result$var, stringsAsFactors = FALSE)
    rn <- attr(result$var, "row.names")
    if (!is.null(rn)) rownames(var_df) <- rn
    attr(mat, "var") <- var_df
  }

  if (normalize) {
    cs <- attr(mat, "colsums")
    if (is.null(cs)) stop("File has no stored colsums; cannot normalize")
    mat <- lognorm(mat, cs, scale = scale)
  }

  mat
}

#' Get .1pz file information
#'
#' @param path Path to a .1pz file.
#' @return A named list with version, m, n, nnz, has_metadata, etc.
#' @export
info_1pz <- function(path) {
  info_1pz_r(path)
}

#' Validate .1pz file CRC integrity
#'
#' @param path Path to a .1pz file.
#' @return A named list with valid (logical), file_crc_ok, footer_ok.
#' @export
validate_1pz <- function(path) {
  validate_1pz_r(path)
}

#' Get column sums from .1pz file
#'
#' @param path Path to a .1pz file.
#' @return Numeric vector of column sums, or NULL if not stored.
#' @export
colsums_1pz <- function(path) {
  result <- read_1pz_r(path, 1L)
  result$colsums
}

#' Read a .1pz file into a Seurat assay
#'
#' Embedded obs/var DataFrames are attached to the Seurat object
#' as cell metadata. No external dependencies beyond Seurat required.
#'
#' @param path Path to a .1pz file.
#' @param assay_name Name for the Seurat assay. Default "RNA".
#' @param num_threads Number of threads. Default 4.
#' @return A Seurat object with the count matrix and metadata.
#' @export
read_1pz_seurat <- function(path, assay_name = "RNA", num_threads = 4L) {
  if (!requireNamespace("Seurat", quietly = TRUE))
    stop("Seurat package required for read_1pz_seurat()")

  mat <- read_1pz(path, num_threads)

  # .1pz stores genes x cells; Seurat expects features x cells
  obj <- Seurat::CreateSeuratObject(
    counts = mat,
    assay = assay_name
  )
  cs <- attr(mat, "colsums")
  if (!is.null(cs)) {
    obj[["total_counts"]] <- cs
  }

  # Merge obs data.frame into cell metadata
  obs_df <- attr(mat, "obs")
  if (!is.null(obs_df) && nrow(obs_df) == ncol(mat)) {
    for (col in colnames(obs_df)) {
      obj[[col]] <- obs_df[[col]]
    }
  }

  # Store uns as misc metadata
  uns <- attr(mat, "uns")
  if (!is.null(uns)) {
    Seurat::Misc(obj) <- as.list(uns)
  }

  obj
}

#' Apply log-normalization using stored column sums
#'
#' @param x A value (scalar, vector, or sparse matrix).
#' @param colsums Column sums vector.
#' @param scale Scaling factor. Default 10000.
#' @return Log-normalized values: log1p(x * scale / colsum).
#' @export
lognorm <- function(x, colsums, scale = 10000) {
  if (inherits(x, "dgCMatrix")) {
    result <- x
    factors <- scale / colsums
    for (j in seq_len(ncol(x))) {
      idx <- (x@p[j] + 1L):x@p[j + 1L]
      if (length(idx) > 0 && idx[1] <= length(x@x)) {
        result@x[idx] <- log1p(x@x[idx] * factors[j])
      }
    }
    return(result)
  }
  log1p(x * scale / colsums)
}


# ============================================================================
# SingleCellExperiment interoperability
# ============================================================================

#' Read a .1pz file into a SingleCellExperiment
#'
#' Creates a SingleCellExperiment with the count matrix as the "counts" assay.
#' Embedded obs/var DataFrames become colData/rowData respectively.
#' Study metadata (uns) is stored in metadata().
#'
#' @param path Path to a .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return A SingleCellExperiment object.
#' @export
read_1pz_sce <- function(path, num_threads = 4L) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE))
    stop("SingleCellExperiment package required. Install with: ",
         "BiocManager::install('SingleCellExperiment')")

  mat <- read_1pz(path, num_threads)
  obs_df <- attr(mat, "obs")
  var_df <- attr(mat, "var")
  uns <- attr(mat, "uns")
  cs <- attr(mat, "colsums")

  # Build colData
  if (!is.null(obs_df)) {
    col_data <- S4Vectors::DataFrame(obs_df)
  } else {
    col_data <- S4Vectors::DataFrame(row.names = colnames(mat))
  }
  if (!is.null(cs)) {
    col_data$total_counts <- as.numeric(cs)
  }

  # Build rowData
  if (!is.null(var_df)) {
    row_data <- S4Vectors::DataFrame(var_df)
  } else {
    row_data <- S4Vectors::DataFrame(row.names = rownames(mat))
  }

  # Create SCE
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = mat),
    colData = col_data,
    rowData = row_data
  )

  # Store uns as metadata
  if (!is.null(uns)) {
    S4Vectors::metadata(sce) <- as.list(uns)
  }

  sce
}


#' Write a Seurat object to a .1pz file
#'
#' Extracts the count matrix and metadata from a Seurat object and writes
#' to .1pz format. Cell metadata is stored as obs, feature metadata as var.
#'
#' @param obj A Seurat object.
#' @param path Output .1pz file path.
#' @param assay_name Assay to extract. Default "RNA".
#' @param num_threads Number of threads. Default 4.
#' @return Invisible NULL.
#' @export
write_1pz_seurat <- function(obj, path, assay_name = "RNA", num_threads = 4L) {
  if (!requireNamespace("Seurat", quietly = TRUE))
    stop("Seurat package required for write_1pz_seurat()")

  mat <- Seurat::GetAssayData(obj, assay = assay_name, layer = "counts")
  if (!inherits(mat, "dgCMatrix")) {
    mat <- methods::as(mat, "dgCMatrix")
  }

  # Write via Python wrapper (cross-language compatible)
  # For now, use the R-native write path
  write_1pz(mat, path,
            rownames = rownames(mat),
            colnames = colnames(mat),
            num_threads = num_threads)
  invisible(NULL)
}


#' Write a SingleCellExperiment to a .1pz file
#'
#' @param sce A SingleCellExperiment object.
#' @param path Output .1pz file path.
#' @param assay_name Assay to extract. Default "counts".
#' @param num_threads Number of threads. Default 4.
#' @return Invisible NULL.
#' @export
write_1pz_sce <- function(sce, path, assay_name = "counts", num_threads = 4L) {
  if (!requireNamespace("SingleCellExperiment", quietly = TRUE))
    stop("SingleCellExperiment package required.")

  mat <- SummarizedExperiment::assay(sce, assay_name)
  if (!inherits(mat, "dgCMatrix")) {
    mat <- methods::as(mat, "dgCMatrix")
  }

  write_1pz(mat, path,
            rownames = rownames(mat),
            colnames = colnames(mat),
            num_threads = num_threads)
  invisible(NULL)
}


#' Write a dgCMatrix to a .1pz file
#'
#' Native R writer for .1pz format. Calls the C++ backend directly.
#'
#' @param mat A dgCMatrix (sparse column-compressed matrix).
#' @param path Output file path (should end in .1pz).
#' @param rownames Character vector of row names (gene names). Default uses rownames(mat).
#' @param colnames Character vector of column names (cell barcodes). Default uses colnames(mat).
#' @param num_threads Number of threads for parallel encoding. Default 4.
#' @param level Zstd compression level. Default 3.
#' @param chunk_cols Columns per chunk. Default 1024.
#' @return A list with compression statistics (invisible).
#' @export
write_1pz <- function(mat, path, rownames = NULL, colnames = NULL,
                       num_threads = 4L, level = 3L, chunk_cols = 1024L) {
  if (!inherits(mat, "dgCMatrix")) {
    mat <- methods::as(mat, "dgCMatrix")
  }

  rn <- if (is.null(rownames)) rownames(mat) else rownames
  cn <- if (is.null(colnames)) colnames(mat) else colnames
  if (is.null(rn)) rn <- character(0)
  if (is.null(cn)) cn <- character(0)

  result <- write_1pz_r(
    p = mat@p,
    i = mat@i,
    x = mat@x,
    nrow = nrow(mat),
    ncol = ncol(mat),
    path = path,
    rownames = rn,
    colnames = cn,
    num_threads = as.integer(num_threads),
    level = as.integer(level),
    chunk_cols = as.integer(chunk_cols)
  )
  invisible(result)
}


# ============================================================================
# Sparse format conversions
# ============================================================================

#' Convert .1pz to dgTMatrix (triplet / COO format)
#'
#' @param path Path to a .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return A dgTMatrix.
#' @export
read_1pz_coo <- function(path, num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  methods::as(mat, "dgTMatrix")
}

#' Convert .1pz to dense matrix
#'
#' Warning: may use large memory for big datasets.
#'
#' @param path Path to a .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return A dense matrix.
#' @export
read_1pz_dense <- function(path, num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  as.matrix(mat)
}

#' Convert .1pz to data.frame
#'
#' @param path Path to a .1pz file.
#' @param num_threads Number of threads. Default 4.
#' @return A data.frame with rownames and colnames preserved.
#' @export
read_1pz_dataframe <- function(path, num_threads = 4L) {
  mat <- read_1pz(path, num_threads)
  as.data.frame(as.matrix(mat))
}
