# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/pseudobulk.R
#
# GPU donor pseudobulk aggregation + DE:
#   aggregate_pseudobulk() — sum counts per (cluster, donor)
#   pseudobulk_de()        — NB GLM DE on pseudobulks
#
# Mirrors muscat::aggregateData(fun="sum") + muscat::pbDS(method="DESeq2").

#' GPU pseudobulk aggregation
#'
#' Aggregates (sums) per-cell counts into \emph{pseudobulk} profiles for each
#' (cluster, donor) combination, mirroring
#' \code{muscat::aggregateData(fun="sum")}.
#'
#' @details
#' For each pair of cluster label and donor/sample label, all cells belonging
#' to that group have their counts summed into a single pseudoreplicate column.
#' Pseudobulks with fewer than \code{min_cells_per_pseudobulk} cells are
#' excluded from the result.
#'
#' The function returns a \strong{new} \code{SingleCellExperiment} where:
#' \itemize{
#'   \item Each \emph{column} is a (cluster, donor) pseudobulk.
#'   \item \code{assays(pb)$counts} holds the summed count matrix.
#'   \item \code{colData(pb)} has columns \code{cluster_id}, \code{donor_id},
#'         and \code{n_cells}.
#' }
#' This output shape is compatible with \code{muscat::pbDS()}.
#'
#' @param sce                      A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param group_by                 Character.  Column in \code{colData(sce)} giving
#'   cell-cluster labels.
#' @param sample_by                Character.  Column in \code{colData(sce)} giving
#'   donor/sample labels.
#' @param layer                    Character.  Input assay.  Default: \code{"counts"}.
#' @param min_cells_per_pseudobulk Integer.  Minimum cells per pseudobulk.
#'   Default: \code{10L}.
#'
#' @return A new \code{SingleCellExperiment} (pseudobulk SCE) with
#'   \code{assays(pb)$counts} and \code{colData(pb)} columns
#'   \code{cluster_id}, \code{donor_id}, \code{n_cells}.
#'   Mirrors \code{muscat::aggregateData(fun="sum")}.
#'
#' @examples
#' \dontrun{
#' pb <- aggregate_pseudobulk(sce,
#'                             group_by  = "leiden",
#'                             sample_by = "donor_id")
#' dim(pb)
#' }
#'
#' @seealso \code{\link{pseudobulk_de}}
#' @export
aggregate_pseudobulk <- function(sce,
                                 group_by,
                                 sample_by,
                                 layer                    = "counts",
                                 min_cells_per_pseudobulk = 10L) {

    .assert_sce(sce, "aggregate_pseudobulk")

    for (col in c(group_by, sample_by)) {
        if (!col %in% names(SummarizedExperiment::colData(sce)))
            stop(sprintf("aggregate_pseudobulk: colData column '%s' not found.", col))
    }
    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("aggregate_pseudobulk: assay '%s' not found.", layer))

    min_cells_per_pseudobulk <- as.integer(min_cells_per_pseudobulk)

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    # Encode cluster and donor labels as 0-based integers
    raw_cluster <- SummarizedExperiment::colData(sce)[[group_by]]
    raw_donor   <- SummarizedExperiment::colData(sce)[[sample_by]]
    n_cells     <- ncol(sce)

    cluster_int <- .encode_0based(raw_cluster)
    donor_int   <- .encode_0based(raw_donor)

    res <- pseudobulk_aggregate_cpp(mat,
                                    cluster_int,
                                    donor_int,
                                    min_cells_per_pseudobulk)

    n_pb <- res$n_pseudobulks
    meta <- res$pseudobulk_meta

    # Recover label names from integer codes
    cluster_levels <- .factor_levels(raw_cluster)
    donor_levels   <- .factor_levels(raw_donor)

    pb_cluster_names <- cluster_levels[meta$cluster_id + 1L]
    pb_donor_names   <- donor_levels  [meta$donor_id   + 1L]

    pb_coldata <- S4Vectors::DataFrame(
        cluster_id = factor(pb_cluster_names, levels = cluster_levels),
        donor_id   = factor(pb_donor_names,   levels = donor_levels),
        n_cells    = meta$n_cells
    )

    pb <- SingleCellExperiment::SingleCellExperiment(
        assays   = list(counts = res$counts),
        rowData  = SummarizedExperiment::rowData(sce),
        colData  = pb_coldata
    )
    rownames(pb) <- rownames(sce)
    colnames(pb) <- paste0(pb_cluster_names, ".", pb_donor_names)

    pb
}


#' GPU donor pseudobulk differential expression
#'
#' Runs negative-binomial GLM DE on donor pseudobulk profiles per cluster,
#' mirroring the output shape of \code{muscat::pbDS(method="DESeq2")}.
#'
#' @details
#' Internally this function calls \code{\link{aggregate_pseudobulk}} to
#' produce pseudobulk profiles and then runs the GPU NB GLM DE kernel per
#' cluster.  When \code{apeglm_shrinkage = TRUE}, log2 fold changes are
#' shrunk using an apeGLM-style adaptive t-prior, reducing noise for genes
#' with low counts or high dispersion.
#'
#' The return value matches \code{muscat::pbDS()} output shape loosely:
#' a named list of \code{data.frame}s, one per cluster, each with columns
#' \code{gene}, \code{log2FC}, \code{pvalue}, \code{padj}, \code{dispersion}.
#'
#' @param sce                      A \code{SingleCellExperiment}.
#' @param group_by                 Character.  Cluster column in \code{colData}.
#' @param sample_by                Character.  Donor/sample column in \code{colData}.
#' @param layer                    Character.  Input assay.  Default: \code{"counts"}.
#' @param min_cells_per_pseudobulk Integer.  Minimum cells per pseudobulk.
#'   Default: \code{10L}.
#' @param apeglm_shrinkage         Logical.  Apply apeGLM-style log2FC shrinkage.
#'   Default: \code{TRUE}.
#' @param seed                     Integer.  RNG seed for NB GLM.  Default: \code{0L}.
#'
#' @return A named \code{list} of \code{data.frame}s, one per cluster, each
#'   with columns \code{gene}, \code{log2FC}, \code{pvalue}, \code{padj},
#'   \code{dispersion}, ordered by \code{padj}.
#'   Mirrors \code{muscat::pbDS(method="DESeq2")} output shape.
#'
#' @examples
#' \dontrun{
#' res <- pseudobulk_de(sce,
#'                      group_by  = "leiden",
#'                      sample_by = "donor_id")
#' head(res[["0"]])  # results for cluster 0
#' }
#'
#' @seealso \code{\link{aggregate_pseudobulk}}
#' @export
pseudobulk_de <- function(sce,
                           group_by,
                           sample_by,
                           layer                    = "counts",
                           min_cells_per_pseudobulk = 10L,
                           apeglm_shrinkage         = TRUE,
                           seed                     = 0L) {

    .assert_sce(sce, "pseudobulk_de")

    for (col in c(group_by, sample_by)) {
        if (!col %in% names(SummarizedExperiment::colData(sce)))
            stop(sprintf("pseudobulk_de: colData column '%s' not found.", col))
    }
    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("pseudobulk_de: assay '%s' not found.", layer))

    min_cells_per_pseudobulk <- as.integer(min_cells_per_pseudobulk)
    seed                     <- as.integer(seed)

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    # Aggregate pseudobulks
    raw_cluster <- SummarizedExperiment::colData(sce)[[group_by]]
    raw_donor   <- SummarizedExperiment::colData(sce)[[sample_by]]

    cluster_int    <- .encode_0based(raw_cluster)
    donor_int      <- .encode_0based(raw_donor)
    n_clusters     <- length(.factor_levels(raw_cluster))
    cluster_levels <- .factor_levels(raw_cluster)

    pb_res <- pseudobulk_aggregate_cpp(mat,
                                       cluster_int,
                                       donor_int,
                                       min_cells_per_pseudobulk)

    if (pb_res$n_pseudobulks < 2L)
        stop("pseudobulk_de: fewer than 2 pseudobulks passed the min_cells filter.")

    pb_mat      <- pb_res$counts
    pb_meta     <- pb_res$pseudobulk_meta
    n_pb        <- pb_res$n_pseudobulks

    pb_cluster  <- as.integer(pb_meta$cluster_id)
    pb_donor    <- as.integer(pb_meta$donor_id)

    # Run DE kernel
    de_res <- donor_pseudobulk_de_cpp(
        pb_mat,
        pb_cluster,
        pb_donor,
        n_clusters,
        as.logical(apeglm_shrinkage),
        seed
    )

    # Build named list of data.frames with gene names, sorted by padj
    gene_names <- rownames(sce)

    out <- vector("list", n_clusters)
    names(out) <- cluster_levels

    for (cl in seq_len(n_clusters)) {
        cl0 <- cl - 1L   # 0-based kernel index
        cr  <- de_res[[as.character(cl0)]]
        if (is.null(cr)) {
            out[[cl]] <- data.frame(
                gene       = character(0),
                log2FC     = numeric(0),
                pvalue     = numeric(0),
                padj       = numeric(0),
                dispersion = numeric(0),
                stringsAsFactors = FALSE
            )
            next
        }

        ng         <- length(cr$log2FC)
        # Replace 0-based gene index strings with actual gene names
        gene_col   <- if (!is.null(gene_names) && length(gene_names) == nrow(mat)) {
            gene_names[seq_len(ng)]
        } else {
            cr$gene
        }

        df <- data.frame(
            gene       = gene_col,
            log2FC     = cr$log2FC,
            pvalue     = cr$pvalue,
            padj       = cr$padj,
            dispersion = cr$dispersion,
            stringsAsFactors = FALSE
        )
        df <- df[order(df$padj, df$pvalue), ]
        rownames(df) <- NULL
        out[[cl]] <- df
    }

    out
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Convert a factor / character / integer vector to 0-based integer
.encode_0based <- function(x) {
    if (is.factor(x)) return(as.integer(x) - 1L)
    if (is.character(x)) {
        lvls <- sort(unique(x))
        return(match(x, lvls) - 1L)
    }
    iv <- as.integer(x)
    iv - min(iv)
}

# Return the unique level labels for a factor / character / integer vector
.factor_levels <- function(x) {
    if (is.factor(x)) return(levels(x))
    if (is.character(x)) return(sort(unique(x)))
    as.character(sort(unique(as.integer(x))))
}
