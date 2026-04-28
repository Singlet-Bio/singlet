# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/ase.R
#
# GPU DAESC-style allele-specific expression (ASE) scanning:
#   run_ase() — beta-binomial ASE scan across single cells
#
# Mirrors the DAESC R package output format.

#' GPU allele-specific expression (ASE) analysis (DAESC-style)
#'
#' Scans for allele-specific expression at heterozygous SNP sites using a
#' GPU-native beta-binomial model (DAESC; Sun et al. 2023).
#'
#' @details
#' For each SNP, Fisher scoring estimates the allelic imbalance parameter
#' \eqn{\beta} (logit-scale allelic ratio) and the overdispersion \eqn{\phi}
#' across cells, controlling for cell-level read depth variation.
#' BH FDR correction is applied per gene or globally.
#'
#' Optional stratified analysis (when \code{cell_type} is provided) repeats
#' the per-type MLE, yielding cell-type-specific ASE estimates.
#'
#' @param sce         A \code{SingleCellExperiment} with \code{colData} giving
#'   optional cell type labels.
#' @param snp_ad      \code{dgCMatrix} [snps × cells] — alt-allele counts
#'   (from \code{snp_ad.1pz}).
#' @param snp_dp      \code{dgCMatrix} [snps × cells] — total depth
#'   (from \code{snp_dp.1pz}).
#' @param cell_type_col Character. Column in \code{colData(sce)} for stratified
#'   ASE.  \code{NULL} for global (unstratified) model.  Default: \code{NULL}.
#' @param chunk_snps  Integer. SNPs per device chunk.  Default: \code{10000L}.
#' @param run_bh_fdr  Logical. Apply BH FDR correction.  Default: \code{TRUE}.
#' @param fdr_alpha   Numeric. BH FDR threshold.  Default: \code{0.05}.
#' @param min_depth   Integer. Minimum depth to include a cell for a SNP.
#'   Default: \code{5L}.
#' @param min_cells   Integer. Minimum qualifying cells per SNP.
#'   Default: \code{5L}.
#' @param seed        Integer. Kept for API symmetry.  Default: \code{0L}.
#' @param resource    Character. \code{"auto"}, \code{"gpu"}, or \code{"cpu"}.
#'   Default: \code{"auto"}.
#'
#' @return A named \code{list} with per-SNP vectors:
#' \describe{
#'   \item{\code{beta}}{Allelic log-odds (MLE estimate).}
#'   \item{\code{se}}{Standard error of beta.}
#'   \item{\code{phi}}{Beta-binomial overdispersion parameter.}
#'   \item{\code{p_value}}{Uncorrected Wald p-value.}
#'   \item{\code{p_adj}}{BH-adjusted q-value (NaN if \code{run_bh_fdr = FALSE}).}
#'   \item{\code{converged}}{Logical, whether Fisher scoring converged.}
#'   \item{\code{n_cells}}{Integer, number of cells used per SNP.}
#'   \item{\code{strat_beta}}{\code{NumericMatrix} [n_snps × n_types] — stratified
#'     estimates (empty if \code{cell_type_col = NULL}).}
#'   \item{\code{strat_se}}{\code{NumericMatrix} [n_snps × n_types].}
#'   \item{\code{strat_phi}}{\code{NumericMatrix} [n_snps × n_types].}
#' }
#'
#' @examples
#' \dontrun{
#' res <- run_ase(sce, snp_ad = ad_mat, snp_dp = dp_mat)
#' sig <- which(res$p_adj < 0.05)
#' cat(length(sig), "ASE SNPs at 5% FDR\n")
#' }
#'
#' @seealso \code{\link{run_eqtl}}, \code{\link{pseudobulk_de}}
#' @export
run_ase <- function(sce,
                    snp_ad,
                    snp_dp,
                    cell_type_col = NULL,
                    chunk_snps    = 10000L,
                    run_bh_fdr    = TRUE,
                    fdr_alpha     = 0.05,
                    min_depth     = 5L,
                    min_cells     = 5L,
                    seed          = 0L,
                    resource      = "auto") {

    .assert_sce(sce, "run_ase")

    if (!methods::is(snp_ad, "dgCMatrix")) snp_ad <- methods::as(snp_ad, "dgCMatrix")
    if (!methods::is(snp_dp, "dgCMatrix")) snp_dp <- methods::as(snp_dp, "dgCMatrix")

    cell_type <- NULL
    if (!is.null(cell_type_col)) {
        if (!cell_type_col %in% names(SummarizedExperiment::colData(sce)))
            stop(sprintf("run_ase: colData column '%s' not found.", cell_type_col))
        cell_type <- SummarizedExperiment::colData(sce)[[cell_type_col]]
    }

    daesc_cpp(
        snp_ad      = snp_ad,
        snp_dp      = snp_dp,
        cell_type   = cell_type,
        chunk_snps  = as.integer(chunk_snps),
        run_bh_fdr  = as.logical(run_bh_fdr),
        fdr_alpha   = as.double(fdr_alpha),
        min_depth   = as.integer(min_depth),
        min_cells   = as.integer(min_cells),
        seed        = as.integer(seed),
        resource    = as.character(resource)
    )
}
