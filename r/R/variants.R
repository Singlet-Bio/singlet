# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/variants.R
#
# GPU Monopogen-style somatic variant calling:
#   call_variants() — germline genotyping + somatic candidate detection
#
# Mirrors the Monopogen Python tool (Liu et al. 2022) output format.

#' GPU single-cell somatic variant calling (Monopogen-style)
#'
#' Calls germline genotypes and detects somatic mutation candidates from
#' single-cell sequencing data using a GPU-native statistical framework
#' (Monopogen; Liu et al. 2022).
#'
#' @details
#' The pipeline:
#' \enumerate{
#'   \item Pileup aggregation: per-SNP alt-allele frequency + quality scores.
#'   \item Germline Bayesian genotyping: P(0/0), P(0/1), P(1/1) per SNP.
#'   \item Somatic candidate filtering: allele frequency in [af_min, af_max] AND
#'         LRT statistic testing for systematic deviation from germline expectations.
#'   \item LD-score correction: per-chromosome LD panel downweights SNPs in
#'         high-LD regions to reduce false-positive somatic calls.
#'   \item BH FDR correction across somatic candidates.
#' }
#'
#' SNPs must be sorted by chromosome (contiguous blocks) in the input matrices.
#' LD panels must be provided for each chromosome represented in the data.
#'
#' @param sce            A \code{SingleCellExperiment} (used only for shape
#'   validation; the actual data comes from \code{snp_ad} and \code{snp_dp}).
#' @param snp_ad         \code{dgCMatrix} [snps × cells] — alt-allele counts
#'   (from \code{snp_ad.1pz}).
#' @param snp_dp         \code{dgCMatrix} [snps × cells] — total depth
#'   (from \code{snp_dp.1pz}).
#' @param chromosome     \code{IntegerVector} [n_snps], 0-based chromosome index.
#'   SNPs must be sorted by chromosome (contiguous blocks).
#' @param ld_panels      Named \code{list} of per-chromosome LD panels.
#'   Each element is a named list with:
#'   \code{n_snps} (integer), \code{indptr} (1-based), \code{col_idx} (1-based),
#'   \code{val} (float LD r-values ∈ [-1, 1]).
#'   Element 1 = chromosome index 0, element 2 = chromosome index 1, etc.
#'   Empty chromosomes should have \code{n_snps = 0} and empty arrays.
#' @param somatic_af_min Numeric. Minimum allele frequency for somatic candidates.
#'   Default: \code{0.05}.
#' @param somatic_af_max Numeric. Maximum allele frequency for somatic candidates.
#'   Default: \code{0.35}.
#' @param fdr_alpha      Numeric. BH FDR threshold.  Default: \code{0.05}.
#' @param run_bh_fdr     Logical. Apply BH FDR.  Default: \code{TRUE}.
#' @param min_total_dp   Integer. Minimum total depth across all cells.
#'   Default: \code{10L}.
#' @param seed           Integer. Kept for API symmetry (algorithm is
#'   deterministic).  Default: \code{0L}.
#' @param resource       Character. \code{"auto"}, \code{"gpu"}, or \code{"cpu"}.
#'   Default: \code{"auto"}.
#'
#' @return A named \code{list}:
#' \describe{
#'   \item{\code{germline_genotypes}}{\code{NumericMatrix} [n_snps × 3] with
#'     columns \code{P_hom_ref}, \code{P_het}, \code{P_hom_alt}.}
#'   \item{\code{germline_calls}}{\code{IntegerVector} [n_snps];
#'     0 = 0/0, 1 = 0/1, 2 = 1/1, -1 = filtered.}
#'   \item{\code{qual_scores}}{\code{NumericVector} [n_snps] (Phred-scaled
#'     posterior quality).}
#'   \item{\code{somatic_candidates}}{data.frame with columns \code{snp_idx}
#'     (1-based), \code{alt_freq}, \code{ld_score}, \code{lrt_stat},
#'     \code{p_value}, \code{q_value}.}
#'   \item{\code{n_somatic}}{Integer, number of somatic candidates.}
#' }
#'
#' @examples
#' \dontrun{
#' res <- call_variants(sce,
#'                      snp_ad    = ad_mat,
#'                      snp_dp    = dp_mat,
#'                      chromosome = snp_chr,
#'                      ld_panels  = hapmap_ld_list)
#' head(res$somatic_candidates)
#' }
#'
#' @seealso \code{\link{run_ase}}, \code{\link{detect_cna}}
#' @export
call_variants <- function(sce,
                          snp_ad,
                          snp_dp,
                          chromosome,
                          ld_panels,
                          somatic_af_min = 0.05,
                          somatic_af_max = 0.35,
                          fdr_alpha      = 0.05,
                          run_bh_fdr     = TRUE,
                          min_total_dp   = 10L,
                          seed           = 0L,
                          resource       = "auto") {

    .assert_sce(sce, "call_variants")

    if (!methods::is(snp_ad, "dgCMatrix")) snp_ad <- methods::as(snp_ad, "dgCMatrix")
    if (!methods::is(snp_dp, "dgCMatrix")) snp_dp <- methods::as(snp_dp, "dgCMatrix")

    chromosome <- as.integer(chromosome)

    if (!is.list(ld_panels))
        stop("call_variants: ld_panels must be a list of per-chromosome LD panels.")

    call_variants_cpp(
        snp_ad        = snp_ad,
        snp_dp        = snp_dp,
        chromosome    = chromosome,
        ld_panels     = ld_panels,
        somatic_af_min = as.double(somatic_af_min),
        somatic_af_max = as.double(somatic_af_max),
        fdr_alpha     = as.double(fdr_alpha),
        run_bh_fdr    = as.logical(run_bh_fdr),
        min_total_dp  = as.integer(min_total_dp),
        seed          = as.integer(seed),
        resource      = as.character(resource)
    )
}
