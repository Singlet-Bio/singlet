# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/rank_genes_groups.R
#
# GPU differential expression: per-cluster marker gene ranking.
#   rank_genes_groups()  — Wilcoxon or t-test DE, mirroring scanpy naming

#' GPU per-cluster differential expression
#'
#' Ranks genes by their differential expression score for each cluster using
#' either the Wilcoxon rank-sum test (histogram-binned, ScaleSC method) or
#' Welch's t-test, both implemented as GPU kernels in singlet-gpu.
#'
#' @details
#' Results are stored in \code{metadata(sce)$rank_genes_groups} as a
#' \code{data.frame} with columns:
#' \describe{
#'   \item{\code{cluster}}{Character, cluster label.}
#'   \item{\code{gene}}{Character, gene name.}
#'   \item{\code{z_score}}{Numeric, Wilcoxon z or Welch t statistic.}
#'   \item{\code{log2_fc}}{Numeric, log2 fold change vs rest.}
#'   \item{\code{p_value}}{Numeric, two-sided p-value.}
#'   \item{\code{p_adj}}{Numeric, BH-adjusted p-value.}
#'   \item{\code{rank}}{Integer, rank within cluster (1 = most significant).}
#' }
#'
#' The function is also invisibly returned for compatibility with pipe-based
#' workflows; the DataFrame is the primary result.
#'
#' \strong{Groupby}: the cluster labels are read from \code{colData(sce)[[groupby]]}.
#' Labels must be coercible to integer; factors and character vectors are
#' converted to 0-based integer codes.
#'
#' @param sce     A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param groupby Character.  Column in \code{colData(sce)} containing cluster
#'   labels (e.g., \code{"leiden"}).
#' @param method  Character.  Test to use: \code{"wilcoxon"} (default) or
#'   \code{"t-test"}.
#' @param n_genes Integer.  Maximum marker genes returned per cluster.
#'   Default: \code{100}.
#' @param layer   Character.  Input assay.  Default: \code{"logcounts"}.
#'
#' @return The input \code{sce} with \code{metadata(sce)$rank_genes_groups}
#'   populated.  The \code{data.frame} is also returned invisibly.
#'
#' @examples
#' \dontrun{
#' sce <- leiden(sce, resolution = 1.0)
#' sce <- rank_genes_groups(sce, groupby = "leiden", method = "wilcoxon")
#' head(metadata(sce)$rank_genes_groups)
#' }
#'
#' @seealso \code{\link{leiden}}, \code{\link{score_genes}}
#' @export
rank_genes_groups <- function(sce,
                               groupby,
                               method  = c("wilcoxon", "t-test"),
                               n_genes = 100L,
                               layer   = "logcounts") {

    .assert_sce(sce, "rank_genes_groups")
    method <- match.arg(method)

    if (!groupby %in% names(SummarizedExperiment::colData(sce))) {
        stop(sprintf("rank_genes_groups: colData column '%s' not found.", groupby))
    }
    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("rank_genes_groups: assay '%s' not found.", layer))
    }

    n_genes <- as.integer(n_genes)
    if (n_genes < 1L) stop("rank_genes_groups: n_genes must be >= 1.")

    # Extract and convert labels to 0-based integer
    raw_labels <- SummarizedExperiment::colData(sce)[[groupby]]
    if (is.factor(raw_labels)) {
        int_labels <- as.integer(raw_labels) - 1L  # factor levels are 1-based
    } else if (is.character(raw_labels)) {
        lvls <- sort(unique(raw_labels))
        int_labels <- match(raw_labels, lvls) - 1L
    } else {
        int_labels <- as.integer(raw_labels)
        # Normalize to 0-based if necessary
        min_lbl <- min(int_labels)
        if (min_lbl != 0L) int_labels <- int_labels - min_lbl
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    # Dispatch to appropriate DE kernel
    per_cluster <- if (method == "wilcoxon") {
        wilcoxon_de_cpp(mat, int_labels, n_genes, deterministic = FALSE)
    } else {
        ttest_de_cpp(mat, int_labels, n_genes, deterministic = FALSE)
    }

    # Stitch per-cluster Lists into a single data.frame
    gene_names <- rownames(sce)
    if (is.factor(raw_labels)) {
        cluster_levels <- levels(raw_labels)
    } else if (is.character(raw_labels)) {
        cluster_levels <- sort(unique(raw_labels))
    } else {
        cluster_levels <- as.character(sort(unique(int_labels)))
    }

    rows <- vector("list", length(per_cluster))
    for (ci in seq_along(per_cluster)) {
        cm    <- per_cluster[[ci]]
        ng    <- length(cm$gene_indices)
        if (ng == 0L) next

        # 0-based gene indices → gene names (gene_names is 1-based R vector)
        g_names <- gene_names[cm$gene_indices + 1L]

        clust_lbl <- if (ci <= length(cluster_levels)) {
            cluster_levels[ci]
        } else {
            as.character(ci - 1L)
        }

        rows[[ci]] <- data.frame(
            cluster  = rep(clust_lbl, ng),
            gene     = g_names,
            z_score  = cm$z_scores,
            log2_fc  = cm$log2_fc,
            p_value  = cm$p_values,
            p_adj    = cm$p_adj,
            rank     = seq_len(ng),
            stringsAsFactors = FALSE
        )
    }

    result_df <- do.call(rbind, rows[!vapply(rows, is.null, logical(1))])
    rownames(result_df) <- NULL

    SummarizedExperiment::metadata(sce)$rank_genes_groups <- result_df

    invisible(sce)
}
