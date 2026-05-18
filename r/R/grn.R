# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/grn.R
#
# GPU GRaNIE-style gene regulatory network inference:
#   granie() — GRN inference from multiome (scATAC + scRNA) data
#
# Mirrors the GRaNIE R package (Gaumondo et al. 2024) output format:
# a data.frame of TF→target edges with combined_score, p_value, fdr.

#' GPU GRaNIE-style gene regulatory network inference
#'
#' Infers a transcription-factor → target-gene regulatory network from paired
#' scRNA + scATAC (multiome) data using GPU-native peak–gene and TF-activity
#' correlation analysis.  Mirrors the GRaNIE R package (Gaumondo et al. 2024).
#'
#' @details
#' The pipeline proceeds in four GPU-native steps:
#' \enumerate{
#'   \item Peak filtering (Welford mean + variance, same kernel as \code{\link{hvg}}).
#'   \item Peak–gene Pearson correlation for all cis-window pairs across cells.
#'   \item TF activity estimation: cuSPARSE SpMM of tf_motif × peaks → tf_activity.
#'   \item TF–target combined score: corr(TF_activity, expression) ×
#'         mean(|r_peak_gene| for peaks where TF binds).
#' }
#' BH FDR correction is applied on device.  Optionally, Leiden community detection
#' groups the GRN into transcriptional modules.
#'
#' @param sce            A \code{SingleCellExperiment} with assay \code{"logcounts"}
#'   (scRNA expression) and assay \code{"peaks"} (scATAC accessibility).
#' @param gex_assay      Character. Name of the scRNA assay in \code{sce}.
#'   Default: \code{"logcounts"}.
#' @param peaks_assay    Character. Name of the scATAC assay in \code{sce}.
#'   Default: \code{"peaks"}.
#' @param tf_motif       Named list with slots \code{row_ptr}, \code{col_idx},
#'   \code{values}, \code{n_rows} (= n_tfs), \code{n_cols} (= n_peaks).
#'   Encodes the TF × peak binary motif occurrence matrix in CSR format.
#'   1-based R indices.  Typically constructed from JASPAR/HOMER annotations.
#' @param peak_gene_pairs An \code{IntegerMatrix} with 2 columns:
#'   \code{peak_idx} (1-based), \code{gene_idx} (1-based), listing cis-window
#'   (peak, gene) pairs.  Build with
#'   \code{GenomicRanges::findOverlaps(peaks, genes + window)}.
#' @param tf_names       \code{CharacterVector} [n_tfs], TF names for labeling.
#'   Optional; if length != n_tfs, numeric indices are used.
#' @param gene_names     \code{CharacterVector} [n_genes], gene names for labeling.
#'   Optional; defaults to \code{rownames(sce)}.
#' @param min_abs_r      Numeric. Minimum |Pearson r| for peak–gene links.
#'   Default: \code{0.3}.
#' @param peak_gene_fdr  Numeric. BH FDR threshold for peak–gene links.
#'   Default: \code{0.05}.
#' @param tf_target_fdr  Numeric. BH FDR threshold for TF–target edges.
#'   Default: \code{0.05}.
#' @param min_peak_mean  Numeric. Minimum peak mean accessibility for filtering.
#'   Default: \code{0.01}.
#' @param min_peak_var   Numeric. Minimum peak variance for filtering.
#'   Default: \code{0.01}.
#' @param run_leiden     Logical. Run Leiden community detection on the GRN.
#'   Default: \code{FALSE}.
#' @param leiden_resolution Numeric. Leiden resolution parameter.
#'   Default: \code{1.0}.
#' @param chunk_size_tfs Integer. TFs processed per GPU chunk in step 4.
#'   Default: \code{100L}.
#' @param seed           Integer. RNG seed.  Default: \code{0L}.
#' @param resource       Character. \code{"auto"}, \code{"gpu"}, or \code{"cpu"}.
#'   Default: \code{"auto"}.
#'
#' @return A named \code{list} with:
#' \describe{
#'   \item{\code{edges}}{data.frame with columns \code{tf}, \code{gene},
#'     \code{combined_score}, \code{pvalue}, \code{fdr}, \code{community}.}
#'   \item{\code{tf_activity}}{\code{NumericMatrix} [n_tfs × n_cells] of
#'     per-TF per-cell activity scores.}
#'   \item{\code{n_edges}}{Integer. Number of edges in the filtered GRN.}
#'   \item{\code{n_peak_gene_links}}{Integer. Peak–gene links passing FDR filter.}
#'   \item{\code{n_communities}}{Integer. Communities found (0 if
#'     \code{run_leiden = FALSE}).}
#' }
#'
#' @examples
#' \dontrun{
#' res <- granie(multiome_sce,
#'               tf_motif       = jaspar_motifs_csr,
#'               peak_gene_pairs = cis_pairs_matrix,
#'               tf_names        = tf_gene_symbols,
#'               tf_target_fdr   = 0.1)
#' head(res$edges)
#' }
#'
#' @seealso \code{\link{run_pca}}, \code{\link{run_nmf}}
#' @export
granie <- function(sce,
                   gex_assay         = "logcounts",
                   peaks_assay       = "peaks",
                   tf_motif,
                   peak_gene_pairs,
                   tf_names          = character(0),
                   gene_names        = rownames(sce),
                   min_abs_r         = 0.3,
                   peak_gene_fdr     = 0.05,
                   tf_target_fdr     = 0.05,
                   min_peak_mean     = 0.01,
                   min_peak_var      = 0.01,
                   run_leiden        = FALSE,
                   leiden_resolution = 1.0,
                   chunk_size_tfs    = 100L,
                   seed              = 0L,
                   resource          = "auto") {

    .assert_sce(sce, "granie")

    if (!gex_assay %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("granie: assay '%s' not found in sce.", gex_assay))
    if (!peaks_assay %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("granie: assay '%s' not found in sce.", peaks_assay))
    if (!is.matrix(peak_gene_pairs) && !inherits(peak_gene_pairs, "matrix"))
        stop("granie: peak_gene_pairs must be an IntegerMatrix with 2 columns.")
    if (!is.list(tf_motif))
        stop("granie: tf_motif must be a named list with slots row_ptr, col_idx, values, n_rows, n_cols.")

    gex_mat   <- SummarizedExperiment::assay(sce, gex_assay)
    peaks_mat <- SummarizedExperiment::assay(sce, peaks_assay)

    if (!methods::is(gex_mat,   "dgCMatrix")) gex_mat   <- methods::as(gex_mat,   "dgCMatrix")
    if (!methods::is(peaks_mat, "dgCMatrix")) peaks_mat <- methods::as(peaks_mat, "dgCMatrix")

    if (is.null(gene_names) || length(gene_names) != nrow(sce))
        gene_names <- character(0)
    if (length(tf_names) == 0)
        tf_names <- character(0)

    granie_cpp(
        gex               = gex_mat,
        peaks             = peaks_mat,
        tf_motif          = tf_motif,
        peak_gene_pairs   = peak_gene_pairs,
        tf_names          = as.character(tf_names),
        gene_names        = as.character(gene_names),
        min_abs_r         = as.double(min_abs_r),
        peak_gene_fdr     = as.double(peak_gene_fdr),
        tf_target_fdr     = as.double(tf_target_fdr),
        min_peak_mean     = as.double(min_peak_mean),
        min_peak_var      = as.double(min_peak_var),
        run_leiden        = as.logical(run_leiden),
        leiden_resolution = as.double(leiden_resolution),
        chunk_size_tfs    = as.integer(chunk_size_tfs),
        seed              = as.integer(seed),
        resource          = as.character(resource)
    )
}
