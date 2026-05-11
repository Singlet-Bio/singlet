# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/cna.R
#
# GPU Numbat-style copy-number alteration (CNA) detection:
#   detect_cna()     — per-cell CNA state calls + clone clustering
#   read_pz_cna_sce() — load expression from singlet output for CNA analysis
#
# Mirrors the Numbat R package (Gao et al. 2024) output format.

#' GPU Numbat-style copy-number alteration detection
#'
#' Detects copy-number alterations in single cells using a per-chromosome
#' rolling-window HMM on log-expression ratios, then clusters cells into
#' CNA-based clones via Leiden clustering.  Mirrors the Numbat R package
#' (Gao et al. 2024).
#'
#' @details
#' The pipeline:
#' \enumerate{
#'   \item Log expression ratio vs reference normal (or per-gene bulk mean).
#'   \item Rolling-window smoothing per chromosome.
#'   \item HMM forward–backward in log-space per cell per chromosome.
#'     States: 0 = loss, 1 = neutral, 2 = gain.
#'   \item Clone clustering: Leiden on per-cell CNA patterns.
#' }
#'
#' @param sce              A \code{SingleCellExperiment}.
#' @param gene_chr         \code{IntegerVector} [n_genes], chromosome (1-based;
#'   23 = X, 24 = Y, 25 = MT).  Typically from \code{rowData(sce)$chr}.
#' @param gene_start       \code{IntegerVector} [n_genes], genomic start
#'   position (bp).  Typically from \code{rowData(sce)$start}.
#' @param ref              Optional \code{NumericVector} [n_genes] of per-gene
#'   reference normal expression (e.g., mean across normal cells).
#'   Default: \code{NULL} (use per-gene sample mean as reference).
#' @param counts_assay     Character. Assay to use.  Default: \code{"logcounts"}.
#' @param smooth_half_win  Integer. Rolling-window half-width; full window =
#'   2 × h + 1.  Must satisfy 2h+1 ≤ 15.  Default: \code{5L}.
#' @param trans_prob       Numeric. HMM off-diagonal transition probability
#'   per segment boundary (0, 0.5).  Default: \code{0.01}.
#' @param segments_per_chr Integer. HMM segments per chromosome.
#'   Default: \code{50L}.
#' @param leiden_resolution Numeric. Leiden resolution for clone clustering.
#'   Default: \code{1.0}.
#' @param knn_k            Integer. kNN k for clone clustering graph.
#'   Default: \code{15L}.
#' @param include_sex_chrs Logical. Include chrX/Y in analysis.
#'   Default: \code{FALSE}.
#' @param seed             Integer. RNG seed for Leiden.  Default: \code{0L}.
#' @param resource         Character. \code{"auto"}, \code{"gpu"}, or
#'   \code{"cpu"}.  Default: \code{"auto"}.
#'
#' @return A named \code{list}:
#' \describe{
#'   \item{\code{cna_states}}{\code{IntegerMatrix} [n_cells × n_total_segments];
#'     0 = loss, 1 = neutral, 2 = gain.}
#'   \item{\code{hmm_posteriors}}{R array [n_cells, n_segs, 3] of HMM
#'     posterior probabilities.}
#'   \item{\code{clone_labels}}{\code{IntegerVector} [n_cells], 0-based Leiden
#'     clone IDs.}
#'   \item{\code{n_clones}}{Integer, number of clones detected.}
#'   \item{\code{leiden_modularity}}{Numeric, Leiden modularity score.}
#'   \item{\code{segments}}{data.frame with columns \code{seg_idx},
#'     \code{chr}, \code{gene_start}, \code{gene_end}.}
#'   \item{\code{n_total_segments}}{Integer, total number of HMM segments.}
#' }
#'
#' @examples
#' \dontrun{
#' res <- detect_cna(sce,
#'                   gene_chr   = rowData(sce)$chr,
#'                   gene_start = rowData(sce)$start)
#' table(res$clone_labels)  # clone sizes
#' }
#'
#' @seealso \code{\link{run_ase}}, \code{\link{leiden}}
#' @export
detect_cna <- function(sce,
                       gene_chr,
                       gene_start,
                       ref               = NULL,
                       counts_assay      = "logcounts",
                       smooth_half_win   = 5L,
                       trans_prob        = 0.01,
                       segments_per_chr  = 50L,
                       leiden_resolution = 1.0,
                       knn_k             = 15L,
                       include_sex_chrs  = FALSE,
                       seed              = 0L,
                       resource          = "auto") {

    .assert_sce(sce, "detect_cna")

    if (!counts_assay %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("detect_cna: assay '%s' not found.", counts_assay))

    gene_chr   <- as.integer(gene_chr)
    gene_start <- as.integer(gene_start)

    if (length(gene_chr) != nrow(sce))
        stop(sprintf("detect_cna: gene_chr length (%d) != nrow(sce) (%d).",
                     length(gene_chr), nrow(sce)))
    if (length(gene_start) != nrow(sce))
        stop(sprintf("detect_cna: gene_start length (%d) != nrow(sce) (%d).",
                     length(gene_start), nrow(sce)))

    expr_mat <- SummarizedExperiment::assay(sce, counts_assay)
    if (!methods::is(expr_mat, "dgCMatrix"))
        expr_mat <- methods::as(expr_mat, "dgCMatrix")

    detect_cna_cpp(
        expr              = expr_mat,
        gene_chr          = gene_chr,
        gene_start        = gene_start,
        ref               = ref,
        smooth_half_win   = as.integer(smooth_half_win),
        trans_prob        = as.double(trans_prob),
        segments_per_chr  = as.integer(segments_per_chr),
        leiden_resolution = as.double(leiden_resolution),
        knn_k             = as.integer(knn_k),
        include_sex_chrs  = as.logical(include_sex_chrs),
        seed              = as.integer(seed),
        resource          = as.character(resource)
    )
}


#' Load singlet expression output into an SCE suitable for CNA detection
#'
#' Reads \code{gene_counts.1pz} (or \code{counts.1pz}) from a singlet output
#' directory into a \code{SingleCellExperiment} with log-normalized expression
#' in assay \code{"logcounts"}.  Provides a convenient entry point for
#' \code{\link{detect_cna}}.
#'
#' @param pz_dir  Character.  Path to a singlet output directory.
#' @param use_gene_counts Logical.  Use \code{gene_counts.1pz} (GeneFull,
#'   exon + intron) instead of \code{counts.1pz} (exon only).
#'   Default: \code{TRUE}.
#'
#' @return A \code{SingleCellExperiment} with assays \code{"counts"} and
#'   \code{"logcounts"}.
#'
#' @seealso \code{\link{detect_cna}}
#' @export
read_pz_cna_sce <- function(pz_dir, use_gene_counts = TRUE) {
    modality <- if (use_gene_counts) "gene" else "exon"
    data     <- load_pz_to_sce_cpp(pz_dir, modality, FALSE)

    sce <- SingleCellExperiment::SingleCellExperiment(
        assays  = list(counts = data$counts),
        rowData = S4Vectors::DataFrame(row.names = as.character(data$genes)),
        colData = S4Vectors::DataFrame(row.names = as.character(data$cells))
    )
    rownames(sce) <- as.character(data$genes)
    colnames(sce) <- as.character(data$cells)

    # Add log1p-normalized assay (target_sum = 10000, same as Seurat default)
    log_mat <- lognorm_cpp(data$counts, target_sum = 1e4)
    SummarizedExperiment::assay(sce, "logcounts") <- log_mat

    sce
}
