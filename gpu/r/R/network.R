# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/network.R
#
# GPU weighted gene co-expression network analysis:
#   run_hdwgcna() — hdWGCNA-style module detection (cycle 46)
#
# Closest published CPU analog: hdWGCNA R package (Morabito et al. Cell Reports 2023).


#' GPU weighted gene co-expression network analysis (hdWGCNA)
#'
#' Detects co-expression modules from single-cell data using GPU-accelerated
#' topological overlap matrix (TOM) computation + dynamic tree cutting.  Mirrors
#' \href{https://github.com/smorabit/hdWGCNA}{hdWGCNA} (Cell Reports 2023).
#'
#' @details
#' The algorithm:
#' \enumerate{
#'   \item Computes the Pearson correlation matrix |r|^beta (power adjacency).
#'   \item Computes the topological overlap matrix (TOM).
#'   \item Applies hierarchical clustering on (1 - TOM) + dynamic tree cutting.
#'   \item Computes module eigengenes (first PC per module).
#'   \item Ranks hub genes per module by kME (module eigengene correlation).
#' }
#'
#' Results are stored as:
#' \itemize{
#'   \item \code{rowData(sce)$hdwgcna_module} — module label per gene (0 = grey/unassigned).
#'   \item \code{reducedDim(sce, "hdwgcna_eigengenes")} — cells × n_modules.
#'   \item \code{metadata(sce)$hdwgcna} — full result list including hub gene table.
#' }
#'
#' @param sce             A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param assay_name      Character.  Default: \code{"logcounts"}.
#' @param soft_power      Integer.  Soft-thresholding power beta.  Default: \code{6L}.
#' @param min_module_size Integer.  Minimum genes per module.  Default: \code{30L}.
#' @param n_top_hub       Integer.  Top hub genes per module to return.  Default: \code{20L}.
#' @param seed            Integer.  Default: \code{0L}.
#' @param resource        Character.  Default: \code{"auto"}.
#'
#' @return The input \code{sce} with module results stored in \code{rowData},
#'   \code{reducedDim}, and \code{metadata}.
#'
#' @export
run_hdwgcna <- function(sce,
                        assay_name      = "logcounts",
                        soft_power      = 6L,
                        min_module_size = 30L,
                        n_top_hub       = 20L,
                        seed            = 0L,
                        resource        = "auto") {

    .assert_sce(sce, "run_hdwgcna")

    if (!assay_name %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_hdwgcna: assay '%s' not found.", assay_name))

    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    res <- hdwgcna_cpp(mat,
                       as.integer(soft_power),
                       as.integer(min_module_size),
                       as.integer(n_top_hub),
                       as.integer(seed),
                       as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_hdwgcna: %s", res$error))

    # Store module assignments in rowData
    rd <- SummarizedExperiment::rowData(sce)
    rd$hdwgcna_module <- res$module_assignment
    SummarizedExperiment::rowData(sce) <- rd

    # Store eigengenes as reducedDim
    if (ncol(res$eigengenes) > 0)
        SingleCellExperiment::reducedDim(sce, "hdwgcna_eigengenes") <- res$eigengenes

    # Store hub gene table in metadata
    hub_df <- as.data.frame(res$hub_genes)
    hub_kme_df <- as.data.frame(res$hub_kme)
    names(hub_df)     <- paste0("hub_rank_", seq_len(ncol(hub_df)))
    names(hub_kme_df) <- paste0("kme_rank_", seq_len(ncol(hub_kme_df)))

    SummarizedExperiment::metadata(sce)$hdwgcna <- list(
        module_assignment = res$module_assignment,
        eigengenes        = res$eigengenes,
        hub_genes         = hub_df,
        hub_kme           = hub_kme_df,
        n_modules         = res$n_modules
    )

    sce
}
