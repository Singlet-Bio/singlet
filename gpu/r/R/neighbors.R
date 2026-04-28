# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/neighbors.R
#
# GPU kNN graph construction.
#   neighbors()  — compute kNN graph and store in metadata(sce)$neighbors

#' GPU k-nearest-neighbour graph
#'
#' Computes a k-nearest-neighbour (kNN) graph from a dimensionality-reduction
#' embedding stored in \code{reducedDim(sce, use_dimred)}, using the
#' singlet-gpu GPU kNN kernel (exact cuBLAS GEMM tiling or HNSW via cuVS for
#' n > 10M cells).
#'
#' @details
#' The result is stored in \code{metadata(sce)$neighbors} as a named list:
#' \describe{
#'   \item{\code{$indices}}{Integer matrix \code{[n_cells × n_neighbors]},
#'     0-based neighbor cell indices (sorted ascending by distance within
#'     each row).}
#'   \item{\code{$distances}}{Numeric matrix \code{[n_cells × n_neighbors]},
#'     corresponding distances.}
#'   \item{\code{$n}}{Integer, number of cells.}
#'   \item{\code{$k}}{Integer, actual k used (may be < n_neighbors if n_cells ≤ k).}
#'   \item{\code{$use_dimred}}{Character, the reducedDim slot used.}
#'   \item{\code{$metric}}{Character, distance metric.}
#' }
#'
#' \strong{Note on storage}: \code{metadata(sce)$neighbors} is not a standard
#' SingleCellExperiment slot.  The scran ecosystem uses \code{colPairs(sce)}
#' for kNN graphs; a future cycle will provide a migration helper.
#'
#' \strong{Host-copy cost}: O(n_cells × n_dims) on input and O(n_cells × k) on
#' output.
#'
#' @param sce         A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param n_neighbors Integer.  Number of nearest neighbours.  Default: \code{15}.
#' @param use_dimred  Character.  The \code{reducedDim} slot to use as the
#'   embedding.  Default: \code{"PCA"}.
#' @param k_within    Deprecated / ignored.  Reserved for future within-batch kNN.
#' @param metric      Character.  Distance metric: \code{"euclidean"} (default),
#'   \code{"cosine"}, or \code{"inner"}.
#' @param seed        Integer.  RNG seed forwarded to the HNSW builder.
#'   Ignored in the exact backend.  Default: \code{0}.
#'
#' @return The input \code{sce} with \code{metadata(sce)$neighbors} populated.
#'
#' @examples
#' \dontrun{
#' sce <- run_pca(sce, n_pcs = 50)
#' sce <- neighbors(sce, n_neighbors = 15, use_dimred = "PCA")
#' str(metadata(sce)$neighbors)
#' }
#'
#' @seealso \code{\link{leiden}}, \code{\link{run_umap}}
#' @export
neighbors <- function(sce,
                      n_neighbors = 15L,
                      use_dimred  = "PCA",
                      k_within    = NULL,
                      metric      = c("euclidean", "cosine", "inner"),
                      seed        = 0L) {

    .assert_sce(sce, "neighbors")
    metric <- match.arg(metric)

    available_dimreds <- SingleCellExperiment::reducedDimNames(sce)
    if (!use_dimred %in% available_dimreds) {
        stop(sprintf(
            "neighbors: reducedDim '%s' not found. Available: %s",
            use_dimred,
            paste(available_dimreds, collapse = ", ")))
    }

    emb <- SingleCellExperiment::reducedDim(sce, use_dimred)
    if (!is.matrix(emb)) {
        emb <- as.matrix(emb)
    }
    storage.mode(emb) <- "double"  # ensure NumericMatrix on C++ side

    n_neighbors <- as.integer(n_neighbors)
    if (n_neighbors < 1L)
        stop("neighbors: n_neighbors must be >= 1.")
    if (n_neighbors >= nrow(emb))
        message(sprintf(
            "neighbors: n_neighbors (%d) >= n_cells (%d); k will be clamped.",
            n_neighbors, nrow(emb)))

    res <- neighbors_cpp(emb,
                         n_neighbors,
                         metric,
                         as.integer(seed))

    # Store in metadata; 0-based indices preserved (document in help)
    SummarizedExperiment::metadata(sce)$neighbors <- list(
        indices   = res$indices,
        distances = res$distances,
        n         = res$n,
        k         = res$k,
        use_dimred = use_dimred,
        metric     = metric
    )

    sce
}
