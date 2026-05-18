# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/enrichment.R
#
# GPU gene-set enrichment:
#   run_gsea()   — fast GSEA (fgsea::fgsea drop-in)
#   run_aucell() — AUCell scoring (AUCell::AUCell_run drop-in)

#' GPU fast gene-set enrichment analysis
#'
#' Runs GPU-accelerated FGSEA on a pre-ranked gene list, mirroring the
#' \code{\link[fgsea]{fgsea}} interface from the Bioconductor fgsea package.
#'
#' @details
#' Enrichment scores (ES) are computed using the standard GSEA rank-walk
#' formula.  Normalised ES (NES) and permutation-based p-values are then
#' computed on-device, with BH adjustment applied to produce \code{padj}.
#'
#' The \code{eps} argument controls the boundary for the p-value approximation
#' via beta-distribution fitting (same semantics as \code{fgsea::fgsea(eps)}).
#' Setting \code{eps = 0} forces exact p-values via full permutation.
#'
#' Gene names not present in \code{pathways} are silently skipped.  Pathways
#' smaller than \code{min_size} or larger than \code{max_size} genes (after
#' restricting to genes in \code{stats}) are excluded from the result.
#'
#' @param stats     Named \code{numeric} vector of per-gene statistics (e.g.
#'   log2FC or \eqn{-\log_{10}(p)} values).  Must have gene names as
#'   \code{names(stats)}.
#' @param pathways  Named \code{list} of character vectors.  Each element
#'   names a gene set and gives its member gene symbols.
#' @param eps       \code{numeric(1)}.  Boundary for p-value beta approximation.
#'   Default: \code{0.0} (exact permutation).
#' @param min_size  \code{integer(1)}.  Minimum pathway size after filtering.
#'   Default: \code{15L}.
#' @param max_size  \code{integer(1)}.  Maximum pathway size.  Default: \code{500L}.
#' @param seed      \code{integer(1)}.  RNG seed for permutations.
#'   Default: \code{42L}.
#' @param n_perm    \code{integer(1)}.  Number of permutations.
#'   Default: \code{1000L}.
#'
#' @return A \code{data.frame} with columns \code{pathway}, \code{ES},
#'   \code{NES}, \code{pval}, \code{padj}, \code{leadingEdge} (a list column
#'   of character vectors), ordered by \code{padj}.  Drop-in for
#'   \code{fgsea::fgsea()}.
#'
#' @examples
#' \dontrun{
#' library(fgsea)
#' data(examplePathways)
#' data(exampleRanks)
#' res <- run_gsea(exampleRanks, examplePathways)
#' head(res[order(res$padj), ])
#' }
#'
#' @seealso \code{\link{run_aucell}}, \code{\link{score_genes}}
#' @export
run_gsea <- function(stats,
                     pathways,
                     eps      = 0.0,
                     min_size = 15L,
                     max_size = 500L,
                     seed     = 42L,
                     n_perm   = 1000L) {

    if (!is.numeric(stats) || is.null(names(stats))) {
        stop("run_gsea: stats must be a named numeric vector.")
    }
    if (!is.list(pathways) || is.null(names(pathways))) {
        stop("run_gsea: pathways must be a named list of character vectors.")
    }

    min_size <- as.integer(min_size)
    max_size <- as.integer(max_size)
    n_perm   <- as.integer(n_perm)
    seed     <- as.integer(seed)
    eps      <- as.double(eps)

    if (min_size < 1L) stop("run_gsea: min_size must be >= 1.")
    if (max_size < min_size) stop("run_gsea: max_size must be >= min_size.")
    if (n_perm < 1L) stop("run_gsea: n_perm must be >= 1.")

    # Ensure stats is a plain named double vector
    stats_vec <- as.double(stats)
    names(stats_vec) <- names(stats)

    res <- fgsea_cpp(stats_vec, pathways, eps, min_size, max_size, n_perm, seed)

    # Sort by padj (matching fgsea::fgsea default)
    res <- res[order(res$padj, res$pval), ]
    rownames(res) <- NULL

    res
}


#' GPU AUCell gene-set activity scoring
#'
#' Computes per-cell AUC scores for a collection of gene sets using the
#' GPU-accelerated AUCell kernel, matching the
#' \code{\link[AUCell]{AUCell_run}} interface from Bioconductor.
#'
#' @details
#' For each cell, genes are ranked by expression.  The AUC is then computed
#' as the fraction of the top \code{aucMaxRank} ranked genes that belong to
#' the gene set (Aibar et al. 2017).
#'
#' If \code{normAUC = TRUE} (default), scores are divided by
#' \code{aucMaxRank} to produce values in \eqn{[0, 1]}.
#'
#' AUC scores are stored in \code{colData(sce)} as one column per gene set,
#' named \code{AUC_{set_name}}.
#'
#' @param sce        A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param gene_sets  Named \code{list} of character vectors (gene set members).
#' @param layer      Character.  Input assay.  Default: \code{"counts"}.
#' @param aucMaxRank \code{integer(1)}.  Number of top-ranked genes used to
#'   compute the AUC.  Default: \code{NULL} (uses \code{ceiling(0.05 * nrow(sce))},
#'   matching \code{AUCell}'s default).
#' @param normAUC    \code{logical(1)}.  Normalise AUC by \code{aucMaxRank}.
#'   Default: \code{TRUE}.
#'
#' @return The input \code{sce} with one \code{colData} column per gene set
#'   named \code{AUC_{set_name}}.  Mirrors
#'   \code{AUCell::AUCell_run()} + \code{AUCell::getAUC()}.
#'
#' @examples
#' \dontrun{
#' gs <- list(HYPOXIA = c("VEGFA", "HIF1A", "LDHA"))
#' sce <- run_aucell(sce, gene_sets = gs)
#' head(colData(sce)$AUC_HYPOXIA)
#' }
#'
#' @seealso \code{\link{run_gsea}}, \code{\link{score_genes}}
#' @export
run_aucell <- function(sce,
                       gene_sets,
                       layer      = "counts",
                       aucMaxRank = NULL,
                       normAUC    = TRUE) {

    .assert_sce(sce, "run_aucell")

    if (!is.list(gene_sets) || is.null(names(gene_sets))) {
        stop("run_aucell: gene_sets must be a named list of character vectors.")
    }
    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("run_aucell: assay '%s' not found.", layer))
    }

    n_genes <- nrow(sce)

    # Default aucMaxRank: ceil(5% of genes) — matches AUCell default
    if (is.null(aucMaxRank)) {
        aucMaxRank <- as.integer(ceiling(0.05 * n_genes))
    }
    aucMaxRank <- as.integer(aucMaxRank)
    if (aucMaxRank < 1L || aucMaxRank > n_genes) {
        stop(sprintf("run_aucell: aucMaxRank must be in [1, n_genes=%d].", n_genes))
    }

    gene_names <- rownames(sce)
    if (is.null(gene_names)) {
        stop("run_aucell: rownames(sce) must be set (gene names required).")
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    # aucell_cpp returns NumericMatrix [n_cells × n_sets] with colnames
    auc_mat <- aucell_cpp(mat, gene_sets, gene_names,
                          aucMaxRank, as.logical(normAUC))

    # Write one colData column per set: AUC_{set_name}
    cd        <- SummarizedExperiment::colData(sce)
    set_names <- colnames(auc_mat)
    for (s_idx in seq_len(ncol(auc_mat))) {
        col_name <- paste0("AUC_", set_names[s_idx])
        cd[[col_name]] <- auc_mat[, s_idx]
    }
    SummarizedExperiment::colData(sce) <- cd

    sce
}
