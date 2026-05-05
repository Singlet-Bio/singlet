# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/qc_new.R
#
# GPU advanced quality control:
#   detect_doublets() — OmniDoublet ensemble doublet detection
#   scrublet_score()  — Scrublet-style doublet scoring (cycle 31 kernel)

#' GPU ensemble doublet detection (OmniDoublet)
#'
#' Detects doublets using an ensemble of three GPU-accelerated methods:
#' (1) simulated-doublet local density (Scrublet-style), (2) k-NN co-embedding
#' score, and (3) a PCA-based classifier.  Analog of
#' \code{scDblFinder::scDblFinder()} or \code{DoubletFinder::doubletFinder()}.
#'
#' @details
#' The ensemble score is the geometric mean of the three per-cell doublet
#' probabilities.  A threshold is set at the \code{expected_rate} quantile of
#' the simulated-doublet score distribution (same logic as Scrublet).
#'
#' Results are stored in \code{colData(sce)$doublet_score} and
#' \code{colData(sce)$is_doublet}.
#'
#' @param sce              A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param layer            Character.  Input assay (logcounts recommended).
#'   Default: \code{"logcounts"}.
#' @param n_pcs            Integer.  Number of PCs used in kNN / classifier.
#'   Default: \code{20L}.
#' @param n_neighbors      Integer.  k for kNN density estimation.
#'   Default: \code{30L}.
#' @param sim_doublet_ratio Numeric.  Ratio of simulated doublets to real cells.
#'   Default: \code{2.0}.
#' @param expected_rate    Numeric.  Prior expected doublet rate.  Default: \code{0.05}.
#' @param seed             Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return The input \code{sce} with:
#'   \describe{
#'     \item{\code{colData(sce)$doublet_score}}{Numeric [0, 1] ensemble score.}
#'     \item{\code{colData(sce)$is_doublet}}{Logical, TRUE = predicted doublet.}
#'     \item{\code{metadata(sce)$doublet_detection}}{List with \code{threshold},
#'       \code{n_doublets_pred}, \code{method = "OmniDoublet"}.}
#'   }
#'   Mirrors \code{scDblFinder::scDblFinder()} output.
#'
#' @examples
#' \dontrun{
#' sce <- detect_doublets(sce, expected_rate = 0.08)
#' table(colData(sce)$is_doublet)
#' }
#'
#' @seealso \code{\link{scrublet_score}}
#' @export
detect_doublets <- function(sce,
                            layer              = "logcounts",
                            n_pcs              = 20L,
                            n_neighbors        = 30L,
                            sim_doublet_ratio  = 2.0,
                            expected_rate      = 0.05,
                            seed               = 0L) {

    .assert_sce(sce, "detect_doublets")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("detect_doublets: assay '%s' not found.", layer))

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    n_pcs             <- as.integer(n_pcs)
    n_neighbors       <- as.integer(n_neighbors)
    sim_doublet_ratio <- as.double(sim_doublet_ratio)
    expected_rate     <- as.double(expected_rate)
    seed              <- as.integer(seed)

    if (n_pcs < 1L)  stop("detect_doublets: n_pcs must be >= 1.")
    if (n_neighbors < 1L) stop("detect_doublets: n_neighbors must be >= 1.")
    if (sim_doublet_ratio <= 0) stop("detect_doublets: sim_doublet_ratio must be > 0.")
    if (expected_rate <= 0 || expected_rate >= 1)
        stop("detect_doublets: expected_rate must be in (0, 1).")

    res <- omni_doublet_score_cpp(
        mat, n_pcs, n_neighbors, sim_doublet_ratio, expected_rate, seed)

    cd <- SummarizedExperiment::colData(sce)
    cd$doublet_score <- res$doublet_score
    cd$is_doublet    <- res$is_doublet
    SummarizedExperiment::colData(sce) <- cd

    SummarizedExperiment::metadata(sce)$doublet_detection <- list(
        method         = "OmniDoublet",
        threshold      = res$threshold,
        n_doublets_pred = res$n_doublets_pred,
        n_pcs          = n_pcs,
        n_neighbors    = n_neighbors,
        expected_rate  = expected_rate
    )

    sce
}


#' GPU Scrublet-style doublet scoring
#'
#' Detects doublets by simulating artificial doublets and comparing real cells
#' to their simulated-doublet neighborhood score, using the GPU-accelerated
#' cycle-31 doublet scoring kernel.
#' Analog of \code{Scrublet::scrub_doublets()} or
#' \code{scDblFinder::scDblFinder(method="scran")}.
#'
#' @details
#' For each real cell the score is the fraction of simulated doublets among its
#' \code{n_neighbors} nearest neighbours in PCA space.  A bimodal threshold
#' separates cells with a doublet-like score from singlets.
#'
#' Results are stored in \code{colData(sce)$scrublet_score} and
#' \code{colData(sce)$predicted_doublet}.
#'
#' @param sce              A \code{SingleCellExperiment}.
#' @param layer            Character.  Input assay (raw counts preferred).
#'   Default: \code{"counts"}.
#' @param n_pcs            Integer.  PCA dimensions.  Default: \code{30L}.
#' @param n_neighbors      Integer.  kNN k.  Default: \code{30L}.
#' @param sim_doublet_ratio Numeric.  Simulated doublets / real cells ratio.
#'   Default: \code{2.0}.
#' @param threshold        Numeric or \code{NA}.  \code{NA} (default) = auto
#'   from bimodality test.
#' @param seed             Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return The input \code{sce} with:
#'   \describe{
#'     \item{\code{colData(sce)$scrublet_score}}{Numeric doublet score [0, 1].}
#'     \item{\code{colData(sce)$predicted_doublet}}{Logical.}
#'     \item{\code{metadata(sce)$scrublet}}{List with \code{threshold}.}
#'   }
#'   Mirrors \code{Scrublet::scrub_doublets()} output.
#'
#' @examples
#' \dontrun{
#' sce <- scrublet_score(sce)
#' table(colData(sce)$predicted_doublet)
#' }
#'
#' @seealso \code{\link{detect_doublets}}
#' @export
scrublet_score <- function(sce,
                           layer              = "counts",
                           n_pcs              = 30L,
                           n_neighbors        = 30L,
                           sim_doublet_ratio  = 2.0,
                           threshold          = NA_real_,
                           seed               = 0L) {

    .assert_sce(sce, "scrublet_score")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("scrublet_score: assay '%s' not found.", layer))

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    n_pcs             <- as.integer(n_pcs)
    n_neighbors       <- as.integer(n_neighbors)
    sim_doublet_ratio <- as.double(sim_doublet_ratio)
    seed              <- as.integer(seed)
    # Pass negative value to C++ to signal auto-threshold
    thresh_cpp        <- if (is.na(threshold)) -1.0 else as.double(threshold)

    if (n_pcs < 1L)  stop("scrublet_score: n_pcs must be >= 1.")
    if (n_neighbors < 1L) stop("scrublet_score: n_neighbors must be >= 1.")

    res <- doublet_score_cpp(
        mat, n_pcs, n_neighbors, sim_doublet_ratio, thresh_cpp, seed)

    cd <- SummarizedExperiment::colData(sce)
    cd$scrublet_score    <- res$doublet_score
    cd$predicted_doublet <- res$predicted_doublet
    SummarizedExperiment::colData(sce) <- cd

    SummarizedExperiment::metadata(sce)$scrublet <- list(
        threshold   = res$threshold,
        n_pcs       = n_pcs,
        n_neighbors = n_neighbors
    )

    sce
}
