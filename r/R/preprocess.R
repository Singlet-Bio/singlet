# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/preprocess.R
#
# GPU-accelerated preprocessing functions:
#   lognorm()  — per-cell log1p normalization
#   hvg()      — highly-variable gene selection

#' GPU log-normalization of a SingleCellExperiment
#'
#' Applies per-cell library-size normalization followed by \code{log1p}
#' transformation using the singlet-gpu GPU kernel.  Results are stored in
#' \code{assays(sce)$logcounts}.
#'
#' @details
#' The kernel computes per-cell library sizes (\eqn{t_j = \sum_i X_{ij}}),
#' optionally divides by a target \eqn{T} (default: median of \eqn{t_j > 0}),
#' then applies \eqn{Y_{ij} = \log_1\!(X_{ij} \cdot T / t_j + 1)} in a single
#' fused GPU pass.
#'
#' Cells with zero total counts are flagged internally and receive all-zero
#' logcounts.  A warning is emitted if any such cells are found.
#'
#' **Host-copy cost**: one O(nnz) host→device copy (input) and one O(nnz)
#' device→host copy (output).  Approximately 100–200 ms for a 10k-cell sample.
#'
#' @param sce        A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param target_sum Numeric scalar.  Library-size normalization target.
#'   \code{NULL} or \code{0} uses the median library size (recommended).
#'   Common alternative: \code{10000} (Seurat default).
#' @param layer      Character scalar.  Name of the input assay.
#'   Default: \code{"counts"}.
#'
#' @return The input \code{sce} with \code{assays(sce)$logcounts} populated
#'   (a \code{Matrix::dgCMatrix}).
#'
#' @examples
#' \dontrun{
#' sce <- read_1pz_sce("/path/to/GSM4037629/")
#' sce <- lognorm(sce, target_sum = 1e4)
#' assayNames(sce)  # "counts" "logcounts"
#' }
#'
#' @seealso \code{\link{hvg}}, \code{\link{run_pca}}
#' @export
lognorm <- function(sce,
                    target_sum = NULL,
                    layer      = "counts") {

    .assert_sce(sce, "lognorm")

    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("lognorm: assay '%s' not found in sce.", layer))
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    # 0 / NULL → let the kernel compute the median
    ts <- if (is.null(target_sum) || target_sum == 0) 0.0 else as.double(target_sum)

    logcounts <- lognorm_cpp(mat, ts)

    # Preserve dimnames
    dimnames(logcounts) <- dimnames(mat)

    SummarizedExperiment::assay(sce, "logcounts") <- logcounts
    sce
}


#' GPU highly-variable gene (HVG) selection
#'
#' Selects the top \code{n_top} highly variable genes using the singlet-gpu
#' GPU kernel.  Two flavors are supported: Seurat v3 normalized dispersion
#' (\code{"seurat_v3"}) and Pearson residuals (\code{"pearson_residuals"}).
#'
#' @details
#' **SeuratV3**: two-pass Welford mean/variance → LOWESS fit on log10(mean) →
#' log10(variance), then clips the normalized variance to \code{[0, 50]}.
#'
#' **PearsonResiduals**: computes the approximate Pearson residuals under a
#' negative-binomial null model with \code{theta = 100}, following Lause et al.
#' (2021).
#'
#' Results are written into \code{rowData(sce)}:
#' \describe{
#'   \item{\code{highly_variable}}{Logical — TRUE for the top \code{n_top} genes.}
#'   \item{\code{highly_variable_rank}}{Integer rank (1 = most variable).}
#'   \item{\code{variances_norm}}{Normalized variance score.}
#'   \item{\code{means}}{Per-gene mean expression.}
#'   \item{\code{variances}}{Per-gene variance.}
#' }
#'
#' @param sce    A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param n_top  Integer.  Number of HVGs to flag.  Default: \code{2000}.
#' @param flavor Character.  \code{"seurat_v3"} or \code{"pearson_residuals"}.
#'   Default: \code{"seurat_v3"}.
#' @param layer  Character.  Input assay name.  Default: \code{"counts"}.
#'
#' @return The input \code{sce} with \code{rowData(sce)} columns populated.
#'
#' @examples
#' \dontrun{
#' sce <- lognorm(sce)
#' sce <- hvg(sce, n_top = 2000, flavor = "seurat_v3")
#' sum(rowData(sce)$highly_variable)  # 2000
#' }
#'
#' @seealso \code{\link{lognorm}}, \code{\link{run_pca}}
#' @export
hvg <- function(sce,
                n_top  = 2000L,
                flavor = c("seurat_v3", "pearson_residuals"),
                layer  = "counts") {

    .assert_sce(sce, "hvg")
    flavor <- match.arg(flavor)

    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("hvg: assay '%s' not found in sce.", layer))
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    n_top <- as.integer(n_top)
    if (n_top < 1L || n_top > nrow(sce)) {
        stop(sprintf("hvg: n_top must be in [1, %d].", nrow(sce)))
    }

    res <- hvg_cpp(mat, n_top, flavor)

    # Build rowData columns
    n_genes  <- nrow(sce)
    hv_flag  <- logical(n_genes)
    hv_rank  <- integer(n_genes)

    # res$indices is 1-based
    hv_flag[res$indices] <- TRUE
    hv_rank[res$indices] <- seq_along(res$indices)

    rd <- SummarizedExperiment::rowData(sce)
    rd$highly_variable      <- hv_flag
    rd$highly_variable_rank <- hv_rank
    rd$variances_norm       <- res$scores[match(seq_len(n_genes),
                                                 res$indices,
                                                 nomatch = NA_integer_)]
    rd$means                <- res$mean
    rd$variances            <- res$var

    SummarizedExperiment::rowData(sce) <- rd
    sce
}
