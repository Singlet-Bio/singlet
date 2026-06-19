# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/enrich.R
#
# GPU advanced enrichment analysis:
#   run_ssgsea()  — single-sample GSEA scores (gsva::gsva drop-in)
#   run_progeny() — PROGENy pathway activity (progeny::progeny drop-in)

#' GPU single-sample GSEA (ssGSEA) scores
#'
#' Computes per-cell single-sample GSEA scores for a collection of gene sets,
#' using the GPU-accelerated ssGSEA kernel.
#' Analog of \code{GSVA::gsva(method="ssgsea")} or
#' \code{escape::enrichIt(method="ssGSEA")}.
#'
#' @details
#' For each cell and each gene set, ssGSEA ranks all genes by expression and
#' computes the area under the rank-sorted enrichment score curve, weighted by
#' \code{alpha}-th power of the ranks.  Setting \code{normalize = TRUE}
#' (default) divides by the gene set size to put scores on a \code{[0, 1]}
#' scale.
#'
#' Scores are stored in \code{colData(sce)} as one column per gene set,
#' named \code{ssGSEA_{set_name}}.
#'
#' @param sce        A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param gene_sets  Named \code{list} of character vectors (gene set members).
#' @param layer      Character.  Input assay.  Default: \code{"logcounts"}.
#' @param alpha      Numeric.  Rank-normalization exponent.  Default: \code{0.25}
#'   (matches the ssGSEA paper, Barbie et al. 2009).
#' @param normalize  Logical.  Divide scores by gene set size.  Default: \code{TRUE}.
#'
#' @return The input \code{sce} with one \code{colData} column per gene set
#'   named \code{ssGSEA_{set_name}}.  Mirrors \code{GSVA::gsva(method="ssgsea")}.
#'
#' @examples
#' \dontrun{
#' hallmarks <- msigdbr::msigdbr(species="Homo sapiens", category="H")
#' gs <- split(hallmarks$gene_symbol, hallmarks$gs_name)
#' sce <- run_ssgsea(sce, gene_sets = gs)
#' head(names(colData(sce)))
#' }
#'
#' @seealso \code{\link{run_progeny}}, \code{\link{run_aucell}}, \code{\link{run_gsea}}
#' @export
run_ssgsea <- function(sce,
                       gene_sets,
                       layer     = "logcounts",
                       alpha     = 0.25,
                       normalize = TRUE) {

    .assert_sce(sce, "run_ssgsea")

    if (!is.list(gene_sets) || is.null(names(gene_sets)))
        stop("run_ssgsea: gene_sets must be a named list of character vectors.")
    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_ssgsea: assay '%s' not found.", layer))

    gene_names <- rownames(sce)
    if (is.null(gene_names))
        stop("run_ssgsea: rownames(sce) must be set (gene names required).")

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    alpha     <- as.double(alpha)
    normalize <- as.logical(normalize)

    score_mat <- ssgsea_cpp(mat, gene_sets, gene_names, alpha, normalize)

    # Write colData columns: ssGSEA_{set_name}
    cd        <- SummarizedExperiment::colData(sce)
    set_names <- colnames(score_mat)
    for (s_idx in seq_len(ncol(score_mat))) {
        col_name <- paste0("ssGSEA_", set_names[s_idx])
        cd[[col_name]] <- score_mat[, s_idx]
    }
    SummarizedExperiment::colData(sce) <- cd

    sce
}


#' GPU PROGENy pathway activity scoring
#'
#' Computes pathway activity scores using the PROGENy linear model approach,
#' mirroring \code{progeny::progeny()} from Bioconductor.
#'
#' @details
#' PROGENy weights capture the transcriptional footprint of 14 cancer
#' signaling pathways (EGFR, MAPK, PI3K, TGFb, WNT, VEGF, TNFa, Trail,
#' Hypoxia, Androgen, JAK-STAT, NFkB, p53, Estrogen).  For each cell, the
#' activity score is the inner product of its expression vector (on the
#' \code{top_n} most significant genes per pathway) with the pathway weight
#' vector.
#'
#' Pathway activity scores are stored in \code{colData(sce)} as one numeric
#' column per pathway, named \code{PROGENy_{pathway_name}}.
#'
#' A default set of human PROGENy weights (14 pathways, top 500 genes) is
#' bundled with the package at \code{system.file("extdata/progeny_weights.rds",
#' package="singletGpu")}.  Supply \code{weight_matrix} explicitly to override
#' (e.g., for mouse weights from the \code{progeny} Bioconductor package).
#'
#' @param sce            A \code{SingleCellExperiment}.
#' @param weight_matrix  Named \code{NumericMatrix} \code{[n_genes x n_pathways]}.
#'   Rows must be gene symbols (matching \code{rownames(sce)}).  If
#'   \code{NULL}, the bundled human weights are used.  Default: \code{NULL}.
#' @param layer          Character.  Input assay.  Default: \code{"logcounts"}.
#' @param top_n          Integer.  Top N genes per pathway (0 = all).
#'   Default: \code{100L}.
#' @param scale          Logical.  z-score scale scores across cells.
#'   Default: \code{TRUE}.
#'
#' @return The input \code{sce} with one \code{colData} column per pathway
#'   named \code{PROGENy_{pathway_name}}.
#'   Mirrors \code{progeny::progeny(scale=TRUE, top=100)}.
#'
#' @examples
#' \dontrun{
#' sce <- run_progeny(sce)
#' colnames(colData(sce))   # includes PROGENy_EGFR, PROGENy_MAPK, etc.
#' }
#'
#' @seealso \code{\link{run_ssgsea}}, \code{\link{run_aucell}}
#' @export
run_progeny <- function(sce,
                        weight_matrix = NULL,
                        layer         = "logcounts",
                        top_n         = 100L,
                        scale         = TRUE) {

    .assert_sce(sce, "run_progeny")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_progeny: assay '%s' not found.", layer))

    gene_names <- rownames(sce)
    if (is.null(gene_names))
        stop("run_progeny: rownames(sce) must be set (gene names required).")

    # Load bundled human weights if not supplied
    if (is.null(weight_matrix)) {
        wt_path <- system.file("extdata", "progeny_weights_human_top500.rds",
                               package = "singletGpu")
        if (!nchar(wt_path) || !file.exists(wt_path))
            stop("run_progeny: bundled PROGENy weights not found. ",
                 "Supply weight_matrix explicitly or install the singletGpu ",
                 "package with extdata.")
        weight_matrix <- readRDS(wt_path)
    }

    if (!is.matrix(weight_matrix))
        stop("run_progeny: weight_matrix must be a numeric matrix with row/col names.")
    if (is.null(rownames(weight_matrix)) || is.null(colnames(weight_matrix)))
        stop("run_progeny: weight_matrix must have rownames (genes) and colnames (pathways).")

    weight_genes   <- rownames(weight_matrix)
    pathway_names  <- colnames(weight_matrix)
    storage.mode(weight_matrix) <- "double"

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    top_n <- as.integer(top_n)
    scale <- as.logical(scale)

    score_mat <- progeny_cpp(mat, gene_names, weight_matrix,
                             weight_genes, pathway_names, top_n, scale)

    # Write colData columns: PROGENy_{pathway_name}
    cd <- SummarizedExperiment::colData(sce)
    for (p_idx in seq_len(ncol(score_mat))) {
        col_name <- paste0("PROGENy_", pathway_names[p_idx])
        cd[[col_name]] <- score_mat[, p_idx]
    }
    SummarizedExperiment::colData(sce) <- cd

    sce
}
