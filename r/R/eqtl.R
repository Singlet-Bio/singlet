# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/eqtl.R
#
# GPU NEBULA-style single-cell eQTL mapping:
#   run_eqtl() — NB mixed-model eQTL scan across cells and donors
#
# Mirrors the nebula R package (He et al. 2021) output format.

#' GPU single-cell eQTL mapping (NEBULA-style)
#'
#' Maps expression quantitative trait loci (eQTLs) from single-cell data using
#' a GPU-native negative-binomial mixed model (NEBULA; He et al. 2021).
#'
#' @details
#' The NEBULA model accounts for within-donor correlation via a random donor
#' effect.  For each (SNP, gene) pair, Fisher scoring estimates:
#' \deqn{\log E[Y_{cg}] = \text{geno}_{d(c)} \cdot \beta_g + X_{d(c)}^T \gamma + u_{d(c)}}
#' where \eqn{u_d \sim N(0, \sigma^2)} is the donor random effect.
#'
#' All computation runs on device.  The only host traffic is the one-time
#' per-chunk result DMA to the in-memory result struct.
#'
#' \strong{Scale warning}: At 1M SNPs × 20k genes, the in-memory result is
#' ~240 GB.  For production scale, use the C++ callback path (not yet exposed
#' in this R binding).  Consider chunking at the R level by subsetting
#' \code{snp_ad}/\code{snp_dp} for genome-wide runs.
#'
#' @param sce         A \code{SingleCellExperiment} with a counts assay and
#'   \code{colData} columns \code{donor_id}.
#' @param snp_ad      \code{dgCMatrix} [snps × cells] — alt-allele counts per
#'   SNP per cell (from \code{snp_ad.1pz}).
#' @param snp_dp      \code{dgCMatrix} [snps × cells] — total depth per SNP
#'   per cell (from \code{snp_dp.1pz}).
#' @param donor_col   Character. Column in \code{colData(sce)} giving donor IDs.
#'   Default: \code{"donor_id"}.
#' @param counts_assay Character. Counts assay name.  Default: \code{"counts"}.
#' @param covariates  Optional \code{NumericMatrix} [n_donors × n_cov] of
#'   donor-level covariates (age, sex, PC1–5; max 7).  Default: \code{NULL}.
#' @param chunk_snps  Integer. SNPs per device chunk.  Default: \code{10000L}.
#' @param chunk_genes Integer. Genes per device chunk.  Default: \code{500L}.
#' @param run_bh_fdr  Logical. Apply BH FDR correction per gene.  Default: \code{TRUE}.
#' @param fdr_alpha   Numeric. BH FDR threshold.  Default: \code{0.05}.
#' @param seed        Integer. Kept for API symmetry; eQTL model is deterministic.
#'   Default: \code{0L}.
#' @param resource    Character. \code{"auto"}, \code{"gpu"}, or \code{"cpu"}.
#'   Default: \code{"auto"}.
#'
#' @return A named \code{list} with flat numeric vectors (snp × gene row-major):
#' \describe{
#'   \item{\code{beta}}{eQTL effect size per (snp, gene) pair.}
#'   \item{\code{se}}{Standard error of beta.}
#'   \item{\code{sigma_sq}}{Random-effect variance per (snp, gene).}
#'   \item{\code{p_value}}{Uncorrected Wald p-value.}
#'   \item{\code{p_adj}}{BH-adjusted q-value (NaN if \code{run_bh_fdr = FALSE}).}
#'   \item{\code{converged}}{Logical, whether Fisher scoring converged.}
#'   \item{\code{n_snps}}{Integer, number of SNPs tested.}
#'   \item{\code{n_genes}}{Integer, number of genes tested.}
#' }
#'
#' @examples
#' \dontrun{
#' res <- run_eqtl(sce, snp_ad = ad_mat, snp_dp = dp_mat,
#'                 donor_col = "donor_id")
#' # Reshape to matrix
#' beta_mat <- matrix(res$beta, nrow = res$n_snps, ncol = res$n_genes)
#' }
#'
#' @seealso \code{\link{pseudobulk_de}}, \code{\link{run_ase}}
#' @export
run_eqtl <- function(sce,
                     snp_ad,
                     snp_dp,
                     donor_col    = "donor_id",
                     counts_assay = "counts",
                     covariates   = NULL,
                     chunk_snps   = 10000L,
                     chunk_genes  = 500L,
                     run_bh_fdr   = TRUE,
                     fdr_alpha    = 0.05,
                     seed         = 0L,
                     resource     = "auto") {

    .assert_sce(sce, "run_eqtl")

    if (!donor_col %in% names(SummarizedExperiment::colData(sce)))
        stop(sprintf("run_eqtl: colData column '%s' not found.", donor_col))
    if (!counts_assay %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_eqtl: assay '%s' not found.", counts_assay))

    counts_mat <- SummarizedExperiment::assay(sce, counts_assay)
    if (!methods::is(counts_mat, "dgCMatrix"))
        counts_mat <- methods::as(counts_mat, "dgCMatrix")
    if (!methods::is(snp_ad, "dgCMatrix")) snp_ad <- methods::as(snp_ad, "dgCMatrix")
    if (!methods::is(snp_dp, "dgCMatrix")) snp_dp <- methods::as(snp_dp, "dgCMatrix")

    donor_id <- SummarizedExperiment::colData(sce)[[donor_col]]

    nebula_cpp(
        counts      = counts_mat,
        snp_ad      = snp_ad,
        snp_dp      = snp_dp,
        donor_id    = donor_id,
        covariates  = covariates,
        chunk_snps  = as.integer(chunk_snps),
        chunk_genes = as.integer(chunk_genes),
        run_bh_fdr  = as.logical(run_bh_fdr),
        fdr_alpha   = as.double(fdr_alpha),
        seed        = as.integer(seed),
        resource    = as.character(resource)
    )
}
