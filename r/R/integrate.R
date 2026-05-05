# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/integrate.R
#
# GPU batch integration:
#   run_harmony() — Harmony embedding correction (harmony::RunHarmony drop-in)
#   run_bbknn()   — Batch-balanced kNN (bbknnR::bbknn drop-in)

#' GPU Harmony batch correction
#'
#' Runs GPU-accelerated Harmony batch integration on a cell embedding,
#' mirroring the \code{harmony::RunHarmony} interface for
#' \code{SingleCellExperiment} objects.
#'
#' @details
#' Harmony iteratively corrects the embedding by:
#' 1. Soft-assigning cells to K mixture components.
#' 2. Computing per-batch correction factors via ridge regression.
#' 3. Applying corrections and updating the embedding.
#' Convergence is assessed by the maximum shift in cluster assignments
#' (scalar device→host copy, ≤4 bytes per iteration — within the allowed
#' exception for scalar convergence checks).
#'
#' The corrected embedding is stored in
#' \code{reducedDim(sce, adjusted_dimred)}.  This is directly usable as
#' input to \code{\link{build_neighbors}} (for neighbor graph computation),
#' \code{\link{run_umap}}, or \code{\link{leiden}}.
#'
#' @param sce            A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param batch_key      Character.  Column in \code{colData(sce)} with batch labels.
#' @param use_dimred     Character.  \code{reducedDim} slot to correct.
#'   Default: \code{"PCA"}.
#' @param adjusted_dimred  Character.  Name for the corrected embedding slot.
#'   Default: \code{"Harmony"}.
#' @param n_clusters     Integer.  Number of Harmony clusters K.  Default: \code{20L}.
#' @param max_iter       Integer.  Maximum outer iterations.  Default: \code{10L}.
#' @param tol            Numeric.  Convergence tolerance.  Default: \code{1e-4}.
#' @param seed           Integer.  RNG seed for cluster initialisation.
#'   Default: \code{0L}.
#'
#' @return The input \code{sce} with the corrected embedding stored in
#'   \code{reducedDim(sce, adjusted_dimred)}.  Mirrors
#'   \code{harmony::RunHarmony(object, group.by.vars=batch_key)}.
#'
#' @examples
#' \dontrun{
#' sce <- run_harmony(sce, batch_key = "donor_id")
#' # Then compute neighbors on corrected embedding:
#' sce <- build_neighbors(sce, use_dimred = "Harmony")
#' }
#'
#' @seealso \code{\link{run_bbknn}}, \code{\link{leiden}}
#' @export
run_harmony <- function(sce,
                        batch_key,
                        use_dimred       = "PCA",
                        adjusted_dimred  = "Harmony",
                        n_clusters       = 20L,
                        max_iter         = 10L,
                        tol              = 1e-4,
                        seed             = 0L) {

    .assert_sce(sce, "run_harmony")

    if (!use_dimred %in% SingleCellExperiment::reducedDimNames(sce)) {
        stop(sprintf(
            "run_harmony: reducedDim '%s' not found. Available: %s",
            use_dimred,
            paste(SingleCellExperiment::reducedDimNames(sce), collapse = ", ")))
    }
    if (!batch_key %in% names(SummarizedExperiment::colData(sce))) {
        stop(sprintf(
            "run_harmony: colData column '%s' not found.", batch_key))
    }

    n_clusters <- as.integer(n_clusters)
    max_iter   <- as.integer(max_iter)
    seed       <- as.integer(seed)
    tol        <- as.double(tol)

    # Extract embedding [n_cells × n_pcs]
    emb <- SingleCellExperiment::reducedDim(sce, use_dimred)
    if (!is.matrix(emb)) emb <- as.matrix(emb)
    storage.mode(emb) <- "double"

    # Encode batch labels as 0-based integers
    raw_batch <- SummarizedExperiment::colData(sce)[[batch_key]]
    if (is.factor(raw_batch)) {
        batch_int <- as.integer(raw_batch) - 1L
    } else if (is.character(raw_batch)) {
        lvls      <- sort(unique(raw_batch))
        batch_int <- match(raw_batch, lvls) - 1L
    } else {
        batch_int <- as.integer(raw_batch)
        min_b     <- min(batch_int)
        if (min_b != 0L) batch_int <- batch_int - min_b
    }

    corrected <- harmony_cpp(emb, batch_int, n_clusters, max_iter, tol, seed)

    # Preserve rownames (cell barcodes)
    rownames(corrected) <- colnames(sce)

    SingleCellExperiment::reducedDim(sce, adjusted_dimred) <- corrected

    sce
}


#' GPU batch-balanced kNN graph
#'
#' Computes a batch-balanced kNN graph using the GPU BBKNN kernel, mirroring
#' \code{bbknnR::bbknn} for \code{SingleCellExperiment} objects.
#'
#' @details
#' BBKNN finds \code{neighbors_within_batch} nearest neighbours per batch
#' separately, then merges the per-batch graphs into a single joint graph.
#' This prevents batch-dominated nearest-neighbour lists without requiring
#' explicit batch correction of the embedding itself.
#'
#' Results are stored in \code{metadata(sce)$neighbors} with sub-elements
#' \code{indices}, \code{distances}, and \code{connectivities} (Gaussian
#' kernel weights), matching the bbknnR / Scanpy neighbour graph convention.
#'
#' @param sce                    A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param batch_key              Character.  Column in \code{colData(sce)}.
#' @param use_dimred             Character.  Embedding slot to use.
#'   Default: \code{"PCA"}.
#' @param neighbors_within_batch Integer.  k per batch.  Default: \code{3L}.
#' @param n_pcs                  Integer or \code{NULL}.  Number of PCs to use
#'   (truncates the embedding).  \code{NULL} = use all.
#' @param metric                 Character.  Distance metric: \code{"euclidean"}
#'   (default) or \code{"cosine"}.
#'
#' @return The input \code{sce} with \code{metadata(sce)$neighbors} set to a
#'   \code{list} with elements \code{indices} (IntegerMatrix, 0-based),
#'   \code{distances} (NumericMatrix), \code{connectivities} (NumericMatrix),
#'   \code{n_batches} (int), \code{k_per_batch} (int).
#'   Mirrors \code{bbknnR::bbknn()}.
#'
#' @examples
#' \dontrun{
#' sce <- run_bbknn(sce, batch_key = "donor_id", use_dimred = "PCA")
#' # Use the bbknn graph directly with leiden:
#' sce <- leiden(sce, use_dimred = NULL,
#'               neighbors_key = "neighbors")
#' }
#'
#' @seealso \code{\link{run_harmony}}, \code{\link{leiden}}
#' @export
run_bbknn <- function(sce,
                      batch_key,
                      use_dimred             = "PCA",
                      neighbors_within_batch = 3L,
                      n_pcs                  = NULL,
                      metric                 = "euclidean") {

    .assert_sce(sce, "run_bbknn")

    if (!use_dimred %in% SingleCellExperiment::reducedDimNames(sce)) {
        stop(sprintf(
            "run_bbknn: reducedDim '%s' not found. Available: %s",
            use_dimred,
            paste(SingleCellExperiment::reducedDimNames(sce), collapse = ", ")))
    }
    if (!batch_key %in% names(SummarizedExperiment::colData(sce))) {
        stop(sprintf("run_bbknn: colData column '%s' not found.", batch_key))
    }

    metric                 <- match.arg(metric, c("euclidean", "cosine"))
    neighbors_within_batch <- as.integer(neighbors_within_batch)
    n_pcs_int              <- if (is.null(n_pcs)) 0L else as.integer(n_pcs)

    emb <- SingleCellExperiment::reducedDim(sce, use_dimred)
    if (!is.matrix(emb)) emb <- as.matrix(emb)
    storage.mode(emb) <- "double"

    # Encode batch labels as 0-based integers
    raw_batch <- SummarizedExperiment::colData(sce)[[batch_key]]
    if (is.factor(raw_batch)) {
        batch_int <- as.integer(raw_batch) - 1L
    } else if (is.character(raw_batch)) {
        lvls      <- sort(unique(raw_batch))
        batch_int <- match(raw_batch, lvls) - 1L
    } else {
        batch_int <- as.integer(raw_batch)
        min_b     <- min(batch_int)
        if (min_b != 0L) batch_int <- batch_int - min_b
    }

    res <- bbknn_cpp(emb, batch_int, neighbors_within_batch, n_pcs_int, metric)

    SummarizedExperiment::metadata(sce)$neighbors <- res

    sce
}
