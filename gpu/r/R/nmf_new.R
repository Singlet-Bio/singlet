# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/nmf_new.R
#
# GPU advanced NMF:
#   run_csi_gep() — Consensus Sparse Identification of Gene Expression Programs

#' GPU Consensus Sparse NMF for gene expression program discovery (CSI-GEP)
#'
#' Discovers stable gene expression programs (GEPs) via multi-run consensus
#' non-negative matrix factorization with L1 regularization.  Analog of
#' \code{cNMF} (Kotliar et al. 2019) or \code{RcppML::nmf} with
#' \code{n_runs} + consensus clustering.
#'
#' @details
#' \code{n_runs} independent NMF factorizations are performed from different
#' random seeds derived from the user-supplied \code{seed}.  The resulting W
#' matrices (gene loadings) are concatenated and cosine-similarity consensus
#' clustering identifies the \code{k} most stable program archetypes.  Only
#' programs whose within-cluster cosine similarity exceeds
#' \code{min_stability} are retained as "stable" (though all \code{k} are
#' returned for downstream analysis).
#'
#' Factor matrices are stored as:
#' \describe{
#'   \item{\code{reducedDim(sce, "CSI_GEP")}}{H transposed: \code{[cells × k]}.}
#'   \item{\code{rowData(sce)$CSI_GEP_loadings}}{W: \code{[genes × k]}.}
#'   \item{\code{metadata(sce)$csi_gep}}{List with \code{stability},
#'     \code{n_stable}, \code{k}, \code{n_runs}, \code{seed}.}
#' }
#'
#' @param sce           A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param k             Integer.  Number of GEPs (factors).  Default: \code{10L}.
#' @param n_runs        Integer.  NMF runs for consensus.  Default: \code{20L}.
#' @param layer         Character.  Input assay (TPM-normalized counts preferred).
#'   Default: \code{"logcounts"}.
#' @param l1            Numeric.  L1 regularization on H (usage matrix).
#'   Default: \code{0.05}.
#' @param tol           Numeric.  Convergence tolerance per run.  Default: \code{1e-4}.
#' @param max_iter      Integer.  Max iterations per run.  Default: \code{1000L}.
#' @param min_stability Numeric.  Minimum cosine similarity for a stable program.
#'   Programs below this threshold are flagged in \code{metadata$csi_gep$stability}.
#'   Default: \code{0.95}.
#' @param seed          Integer.  Base RNG seed; each run gets seed+i.
#'   Default: \code{0L}.
#'
#' @return The input \code{sce} with:
#'   \describe{
#'     \item{\code{reducedDim(sce, "CSI_GEP")}}{Matrix [cells × k] — cell usage.}
#'     \item{\code{rowData(sce)$CSI_GEP_loadings}}{Matrix [genes × k] — gene loadings.}
#'     \item{\code{metadata(sce)$csi_gep}}{List with \code{stability} (NumericVector,
#'       one per program), \code{n_stable} (int), \code{k}, \code{n_runs}, \code{seed}.}
#'   }
#'   Mirrors the output of \code{cNMF.consensus()} / \code{RcppML::nmf()}.
#'
#' @examples
#' \dontrun{
#' sce <- run_csi_gep(sce, k = 15L, n_runs = 30L)
#' dim(reducedDim(sce, "CSI_GEP"))   # ncol(sce) × 15
#' meta <- metadata(sce)$csi_gep
#' cat("Stable programs:", meta$n_stable, "/", meta$k, "\n")
#' }
#'
#' @seealso \code{\link{run_nmf}}
#' @export
run_csi_gep <- function(sce,
                        k             = 10L,
                        n_runs        = 20L,
                        layer         = "logcounts",
                        l1            = 0.05,
                        tol           = 1e-4,
                        max_iter      = 1000L,
                        min_stability = 0.95,
                        seed          = 0L) {

    .assert_sce(sce, "run_csi_gep")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_csi_gep: assay '%s' not found.", layer))

    k             <- as.integer(k)
    n_runs        <- as.integer(n_runs)
    max_iter      <- as.integer(max_iter)
    seed          <- as.integer(seed)
    l1            <- as.double(l1)
    tol           <- as.double(tol)
    min_stability <- as.double(min_stability)

    if (k < 1L)       stop("run_csi_gep: k must be >= 1.")
    if (n_runs < 1L)  stop("run_csi_gep: n_runs must be >= 1.")
    if (l1 < 0)       stop("run_csi_gep: l1 must be >= 0.")
    if (min_stability < 0 || min_stability > 1)
        stop("run_csi_gep: min_stability must be in [0, 1].")

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    res <- csi_gep_cpp(mat, k, n_runs, l1, tol, max_iter, min_stability, seed)

    actual_k <- ncol(res$W)

    # W: genes × k
    W_mat <- res$W
    rownames(W_mat) <- rownames(sce)
    colnames(W_mat) <- paste0("GEP", seq_len(actual_k))

    # usage (H transposed): cells × k — standard reducedDim convention
    usage_mat <- res$usage
    rownames(usage_mat) <- colnames(sce)
    colnames(usage_mat) <- paste0("GEP", seq_len(actual_k))

    SingleCellExperiment::reducedDim(sce, "CSI_GEP") <- usage_mat

    rd <- SummarizedExperiment::rowData(sce)
    rd$CSI_GEP_loadings <- W_mat
    SummarizedExperiment::rowData(sce) <- rd

    SummarizedExperiment::metadata(sce)$csi_gep <- list(
        stability = res$stability,
        n_stable  = res$n_stable,
        k         = actual_k,
        n_runs    = n_runs,
        seed      = seed,
        l1        = l1,
        min_stability = min_stability
    )

    sce
}
