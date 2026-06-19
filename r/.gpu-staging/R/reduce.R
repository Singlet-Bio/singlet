# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/reduce.R
#
# GPU dimensionality reduction:
#   run_pca()  — PCA via factornet auto-select SVD
#   run_nmf()  — NMF via factornet MU/CD/Cholesky

#' GPU PCA on a SingleCellExperiment
#'
#' Computes principal components using the factornet GPU SVD suite via
#' singlet-gpu's auto-select adapter.  The routing table is owned by factornet:
#' k < 32 → Lanczos, 32 ≤ k < 64 → Randomized, k ≥ 64 → IRLBA.
#'
#' @details
#' Cell embeddings (V, n_cells × n_pcs) are stored in
#' \code{reducedDim(sce, "PCA")}.  Gene loadings (U, n_genes × n_pcs) are
#' stored in \code{rowData(sce)$PCA_loadings}.  Singular values are stored in
#' \code{metadata(sce)$pca$sdev}.
#'
#' Implicit centering (\eqn{A(v) = Xv - \mu (1^T v)}) is applied when
#' \code{zero_center = TRUE} (default).  The mean vector is never materialized
#' — it is computed inside factornet's GPU kernel.
#'
#' **Host-copy cost**: O(nnz) on input plus O(n_cells × n_pcs + n_genes × n_pcs)
#' on output.
#'
#' @param sce        A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param n_pcs      Integer.  Number of principal components.  Default: \code{50}.
#' @param layer      Character.  Input assay (typically \code{"logcounts"}).
#'   Default: \code{"logcounts"}.
#' @param svd_solver Character.  \code{"auto"} (default) uses factornet's routing
#'   table.  Reserved for future explicit backend selection.
#' @param zero_center Logical.  Apply implicit mean-centering.  Default: \code{TRUE}.
#' @param seed       Integer.  RNG seed for stochastic SVD backends.  Default: \code{0}.
#'
#' @return The input \code{sce} with:
#'   \describe{
#'     \item{\code{reducedDim(sce, "PCA")}}{Matrix [cells × n_pcs] of cell embeddings.}
#'     \item{\code{rowData(sce)$PCA_loadings}}{Matrix [genes × n_pcs] of gene loadings.}
#'     \item{\code{metadata(sce)$pca}}{List with \code{sdev} (singular values),
#'       \code{n_pcs}, \code{svd_solver}, \code{zero_center}, \code{seed}.}
#'   }
#'
#' @examples
#' \dontrun{
#' sce <- lognorm(sce)
#' sce <- run_pca(sce, n_pcs = 50)
#' dim(reducedDim(sce, "PCA"))  # ncol(sce) × 50
#' }
#'
#' @seealso \code{\link{run_nmf}}, \code{\link{lognorm}}
#' @export
run_pca <- function(sce,
                    n_pcs       = 50L,
                    layer       = "logcounts",
                    svd_solver  = "auto",
                    zero_center = TRUE,
                    seed        = 0L) {

    .assert_sce(sce, "run_pca")

    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("run_pca: assay '%s' not found in sce.", layer))
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    n_pcs <- as.integer(n_pcs)
    if (n_pcs < 1L) stop("run_pca: n_pcs must be >= 1.")

    res <- pca_cpp(mat,
                   n_pcs,
                   isTRUE(zero_center),
                   as.integer(seed))

    k <- res$k  # components actually computed (may be < n_pcs if matrix is small)

    # V: cells × k — standard "PCA" embedding (right singular vectors)
    V_mat <- res$V  # n_cells × k, col-major NumericMatrix
    rownames(V_mat) <- colnames(sce)
    colnames(V_mat) <- paste0("PC", seq_len(k))

    # U: genes × k — loadings
    U_mat <- res$U  # n_genes × k
    rownames(U_mat) <- rownames(sce)
    colnames(U_mat) <- paste0("PC", seq_len(k))

    # Store embeddings in reducedDim
    SingleCellExperiment::reducedDim(sce, "PCA") <- V_mat

    # Store gene loadings in rowData
    rd <- SummarizedExperiment::rowData(sce)
    rd$PCA_loadings <- U_mat
    SummarizedExperiment::rowData(sce) <- rd

    # Store provenance in metadata
    SummarizedExperiment::metadata(sce)$pca <- list(
        sdev        = res$d,
        n_pcs       = k,
        svd_solver  = svd_solver,
        zero_center = isTRUE(zero_center),
        seed        = seed
    )

    sce
}


#' GPU NMF on a SingleCellExperiment
#'
#' Factorizes the count (or logcount) matrix into \code{W} (gene programme
#' matrix) and \code{H} (cell usage matrix) using the factornet GPU NMF adapter.
#'
#' @details
#' The MSE loss path runs fully GPU-resident (no host round-trips during
#' iteration).  KL and NB loss paths use host-mediated IRLS per column, which
#' is 50–100x slower than MSE.  This is upstream factornet behavior — singlet-gpu
#' does not modify the loss kernels.
#'
#' Factor matrices are stored as:
#' \describe{
#'   \item{\code{reducedDim(sce, "NMF")}}{H transposed: \code{[cells × k]}.}
#'   \item{\code{rowData(sce)$NMF_loadings}}{W: \code{[genes × k]}.}
#'   \item{\code{metadata(sce)$nmf}}{List with \code{d}, \code{k},
#'     \code{iterations}, \code{converged}, \code{loss}, \code{seed}.}
#' }
#'
#' @param sce       A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param n_factors Integer.  Number of NMF factors.  Default: \code{20}.
#' @param loss      Character.  Loss function: \code{"MSE"}, \code{"KL"}, or
#'   \code{"NB"} (Negative Binomial).  Default: \code{"MSE"}.
#' @param layer     Character.  Input assay.  Default: \code{"logcounts"}.
#' @param seed      Integer.  RNG seed.  Default: \code{0}.
#'
#' @return The input \code{sce} with reducedDim + rowData + metadata updated.
#'
#' @examples
#' \dontrun{
#' sce <- lognorm(sce)
#' sce <- run_nmf(sce, n_factors = 20, loss = "MSE")
#' dim(reducedDim(sce, "NMF"))   # ncol(sce) × 20
#' }
#'
#' @seealso \code{\link{run_pca}}, \code{\link{lognorm}}
#' @export
run_nmf <- function(sce,
                    n_factors = 20L,
                    loss      = c("MSE", "KL", "NB"),
                    layer     = "logcounts",
                    seed      = 0L) {

    .assert_sce(sce, "run_nmf")
    loss <- match.arg(loss)

    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("run_nmf: assay '%s' not found in sce.", layer))
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    n_factors <- as.integer(n_factors)
    if (n_factors < 1L) stop("run_nmf: n_factors must be >= 1.")

    if (loss %in% c("KL", "NB")) {
        message(sprintf(
            "run_nmf: loss='%s' uses host-mediated IRLS (factornet behavior). ",
            loss), "Expect 50-100x slower than MSE.")
    }

    res <- nmf_cpp(mat, n_factors, loss, as.integer(seed))

    k <- res$k

    # H: n_factors × n_cells (col-major from C++).
    # Transpose to cells × k for standard SCE reducedDim convention.
    H_t <- t(res$H)   # n_cells × k
    rownames(H_t) <- colnames(sce)
    colnames(H_t) <- paste0("NMF", seq_len(k))

    # W: n_genes × k
    W_mat <- res$W
    rownames(W_mat) <- rownames(sce)
    colnames(W_mat) <- paste0("NMF", seq_len(k))

    SingleCellExperiment::reducedDim(sce, "NMF") <- H_t

    rd <- SummarizedExperiment::rowData(sce)
    rd$NMF_loadings <- W_mat
    SummarizedExperiment::rowData(sce) <- rd

    SummarizedExperiment::metadata(sce)$nmf <- list(
        d          = res$d,
        k          = k,
        iterations = res$iterations,
        converged  = res$converged,
        loss       = loss,
        seed       = seed
    )

    sce
}
