# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/abundance.R
#
# GPU differential abundance testing:
#   run_milo() — Milo-style kNN neighborhood DA testing (cycle 47)
#
# Closest published CPU analog: miloR R package (Dann et al. Nat Biotech 2022).


#' GPU differential abundance testing (Milo)
#'
#' Tests for differential cell-state abundance between conditions using
#' GPU-accelerated kNN neighborhood construction + batched negative-binomial
#' GLM.  Mirrors \href{https://github.com/MarioniLab/miloR}{Milo} / miloR
#' (Nature Biotechnology 2022).
#'
#' @details
#' Unlike cycle 9/15 differential expression (tests gene expression),
#' Milo tests whether the \emph{number of cells} in a neighbourhood differs
#' between conditions — sensitive to rare-cell-state shifts where DE is underpowered.
#'
#' The algorithm:
#' \enumerate{
#'   \item Builds a 30-NN graph on the PCA embedding.
#'   \item Randomly samples \code{n_nh_cells} cells as neighbourhood index cells.
#'   \item Counts cells per donor in each neighbourhood.
#'   \item Fits a negative-binomial GLM per neighbourhood (condition ~ donor).
#'   \item Applies graph-weighted FDR correction.
#' }
#'
#' Results are stored as:
#' \itemize{
#'   \item \code{metadata(sce)$milo} — full result list.
#' }
#'
#' @param sce         A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param assay_name  Character.  Default: \code{"logcounts"}.
#' @param donor_col   Character.  \code{colData} column with donor IDs.
#'   Default: \code{"donor_id"}.
#' @param condition_col Character.  \code{colData} column with per-cell condition.
#'   Default: \code{"condition"}.  Values are coerced to a 0-based binary integer
#'   per unique donor (majority vote when a donor has cells in multiple conditions).
#' @param n_neighbors Integer.  kNN graph k.  Default: \code{30L}.
#' @param n_nh_cells  Integer.  Neighbourhood sample size.  \code{0L} = auto
#'   (10\% of cells).  Default: \code{0L}.
#' @param seed        Integer.  Default: \code{0L}.
#' @param resource    Character.  Default: \code{"auto"}.
#'
#' @return The input \code{sce} with \code{metadata(sce)$milo} populated.
#'
#' @export
run_milo <- function(sce,
                     assay_name    = "logcounts",
                     donor_col     = "donor_id",
                     condition_col = "condition",
                     n_neighbors   = 30L,
                     n_nh_cells    = 0L,
                     seed          = 0L,
                     resource      = "auto") {

    .assert_sce(sce, "run_milo")

    if (!assay_name %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_milo: assay '%s' not found.", assay_name))
    if (!donor_col %in% names(SummarizedExperiment::colData(sce)))
        stop(sprintf("run_milo: colData column '%s' not found.", donor_col))
    if (!condition_col %in% names(SummarizedExperiment::colData(sce)))
        stop(sprintf("run_milo: colData column '%s' not found.", condition_col))

    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    # Encode donor IDs as 0-based integers
    don_raw  <- SummarizedExperiment::colData(sce)[[donor_col]]
    don_factor <- as.factor(don_raw)
    donor_ids  <- as.integer(don_factor) - 1L
    n_donors   <- nlevels(don_factor)

    # Encode per-donor condition via majority vote
    cond_raw   <- SummarizedExperiment::colData(sce)[[condition_col]]
    cond_factor <- as.factor(cond_raw)
    cond_int    <- as.integer(cond_factor) - 1L

    # Majority vote per donor → integer condition [0, n_conditions-1]
    conditions <- integer(n_donors)
    for (d in seq_len(n_donors) - 1L) {
        mask  <- donor_ids == d
        tab   <- table(cond_int[mask])
        conditions[d + 1L] <- as.integer(names(tab)[which.max(tab)])
    }

    res <- milo_cpp(mat,
                    as.integer(donor_ids),
                    as.integer(conditions),
                    as.integer(n_neighbors),
                    as.integer(n_nh_cells),
                    as.integer(seed),
                    as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_milo: %s", res$error))

    SummarizedExperiment::metadata(sce)$milo <- list(
        log_fc          = res$log_fc,
        se              = res$se,
        p_value         = res$p_value,
        fdr             = res$fdr,
        nh_counts       = res$nh_counts,
        n_neighborhoods = res$n_neighborhoods,
        donor_levels    = levels(don_factor),
        condition_levels = levels(cond_factor)
    )

    sce
}
