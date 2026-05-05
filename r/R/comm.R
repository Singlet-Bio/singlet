# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/comm.R
#
# GPU cell-cell communication inference:
#   run_cellchat() — CellChat-style L-R permutation test (cycle 37)
#
# Closest published CPU analog: CellChat R package (Jin et al. Nat Protocols 2024).


#' GPU cell-cell communication inference (CellChat)
#'
#' Infers ligand-receptor communication probabilities between cell types using
#' GPU-accelerated permutation testing.  Mirrors
#' \href{https://github.com/jinworks/CellChat}{CellChat} (Nat Comm 2024) but
#' runs the K=1000 permutation test entirely on GPU.
#'
#' @details
#' The algorithm:
#' \enumerate{
#'   \item Computes per-cell-type mean expression for every ligand and receptor gene.
#'   \item Applies the Hill-function coupling model for each L-R pair across all
#'     sender-receiver cell-type combinations.
#'   \item Runs K permutations of cell-type labels to build a null distribution and
#'     derive empirical p-values.
#'   \item Aggregates probabilities and p-values by signalling pathway.
#' }
#'
#' Results are stored in \code{metadata(sce)$cellchat}:
#' \describe{
#'   \item{\code{comm_prob}}{NumericMatrix [n_types² × n_lr] communication probabilities.}
#'   \item{\code{p_values}}{NumericMatrix [n_types² × n_lr] permutation p-values.}
#'   \item{\code{pathway_activity}}{NumericMatrix [n_types² × n_pathways] aggregated activity.}
#'   \item{\code{n_types}, \code{n_lr}, \code{n_pathways}}{Dimension metadata.}
#' }
#'
#' @param sce           A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param assay_name    Character.  Default: \code{"logcounts"}.
#' @param cell_type_col Character.  \code{colData} column with cell-type labels.
#'   Default: \code{"cell_type"}.
#' @param lr_database   A \code{data.frame} with columns:
#'   \describe{
#'     \item{\code{ligand}}{Comma-separated gene name(s) (complex allowed).}
#'     \item{\code{receptor}}{Comma-separated gene name(s).}
#'     \item{\code{pathway}}{Signalling pathway name.}
#'   }
#' @param n_permutations Integer.  Permutation count.  Default: \code{1000L}.
#' @param seed           Integer.  Default: \code{0L}.
#' @param resource       Character.  Default: \code{"auto"}.
#'
#' @return The input \code{sce} with \code{metadata(sce)$cellchat} populated.
#'
#' @export
run_cellchat <- function(sce,
                         assay_name    = "logcounts",
                         cell_type_col = "cell_type",
                         lr_database,
                         n_permutations = 1000L,
                         seed           = 0L,
                         resource       = "auto") {

    .assert_sce(sce, "run_cellchat")

    if (!assay_name %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_cellchat: assay '%s' not found.", assay_name))
    if (!cell_type_col %in% names(SummarizedExperiment::colData(sce)))
        stop(sprintf("run_cellchat: colData column '%s' not found.", cell_type_col))
    if (!all(c("ligand", "receptor", "pathway") %in% names(lr_database)))
        stop("run_cellchat: lr_database must have columns: ligand, receptor, pathway.")

    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    gene_names <- rownames(sce)
    if (is.null(gene_names))
        stop("run_cellchat: sce must have rownames (gene names).")

    # Encode cell types as 0-based integers
    ct_raw <- SummarizedExperiment::colData(sce)[[cell_type_col]]
    if (is.factor(ct_raw)) {
        ct_ids <- as.integer(ct_raw) - 1L
    } else {
        ct_factor <- as.factor(ct_raw)
        ct_ids    <- as.integer(ct_factor) - 1L
    }

    # Build L-R index matrices [n_lr × max_set_size]
    n_lr <- nrow(lr_database)

    .parse_gene_set <- function(genes_str, gene_names) {
        parts <- trimws(strsplit(as.character(genes_str), ",")[[1]])
        idx   <- match(parts, gene_names)
        idx[is.na(idx)] <- -1L   # -1 = pad / not found
        idx
    }

    lig_sets <- lapply(lr_database$ligand,   .parse_gene_set, gene_names)
    rec_sets <- lapply(lr_database$receptor, .parse_gene_set, gene_names)
    max_set  <- max(sapply(c(lig_sets, rec_sets), length), 1L)

    lig_mat <- matrix(-1L, nrow = n_lr, ncol = max_set)
    rec_mat <- matrix(-1L, nrow = n_lr, ncol = max_set)
    for (i in seq_len(n_lr)) {
        ls <- lig_sets[[i]]; lig_mat[i, seq_along(ls)] <- ls
        rs <- rec_sets[[i]]; rec_mat[i, seq_along(rs)] <- rs
    }

    # Pathway encoding
    pw_factor  <- as.factor(lr_database$pathway)
    pw_ids     <- as.integer(pw_factor) - 1L
    n_pathways <- nlevels(pw_factor)

    res <- cellchat_cpp(mat,
                        as.integer(ct_ids),
                        lig_mat, rec_mat,
                        as.integer(pw_ids),
                        as.integer(n_pathways),
                        as.integer(n_permutations),
                        as.integer(seed),
                        as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_cellchat: %s", res$error))

    SummarizedExperiment::metadata(sce)$cellchat <- list(
        comm_prob        = res$comm_prob,
        p_values         = res$p_values,
        pathway_activity = res$pathway_activity,
        n_types          = res$n_types,
        n_lr             = res$n_lr,
        n_pathways       = res$n_pathways,
        cell_type_levels = levels(as.factor(ct_raw)),
        pathway_levels   = levels(pw_factor)
    )

    sce
}
