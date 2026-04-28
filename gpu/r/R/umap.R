# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/umap.R
#
# GPU UMAP dimensionality reduction.
#   run_umap()  — compute UMAP embedding and store in reducedDim(sce, "UMAP")

#' GPU UMAP dimensionality reduction
#'
#' Computes a UMAP embedding from the precomputed kNN graph stored in
#' \code{metadata(sce)$neighbors}, using the singlet-gpu cuML UMAP adapter
#' (precomputed-kNN path, no redundant kNN rebuild).
#'
#' @details
#' The embedding is stored in \code{reducedDim(sce, "UMAP")} as a numeric
#' matrix \code{[n_cells × n_components]}.
#'
#' \strong{Prerequisite}: \code{metadata(sce)$neighbors} must exist (call
#' \code{neighbors(sce)} first).
#'
#' \strong{Determinism}: \code{init = "random"} with a fixed \code{seed} is
#' deterministic.  \code{init = "spectral"} is NOT deterministic due to an
#' upstream cuML limitation (cuML GitHub #6696).
#'
#' \strong{Host-copy cost}: O(n_cells × k) on input, O(n_cells × n_components)
#' on output.
#'
#' @param sce          A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param n_components Integer.  Number of UMAP dimensions.  Default: \code{2}.
#' @param min_dist     Numeric.  Minimum distance between points in the embedding.
#'   Default: \code{0.5}.
#' @param spread       Numeric.  Effective scale of embedded points.
#'   Default: \code{1.0}.
#' @param init         Character.  Embedding initialisation: \code{"random"}
#'   (default, deterministic with fixed seed) or \code{"spectral"}
#'   (non-deterministic — see Details).
#' @param seed         Integer.  RNG seed for cuML SGD.  Default: \code{0}.
#' @param use_neighbors Logical.  If \code{TRUE} (default), read the kNN graph
#'   from \code{metadata(sce)$neighbors}.  Reserved for future direct-embedding
#'   mode (not yet implemented).
#'
#' @return The input \code{sce} with:
#'   \describe{
#'     \item{\code{reducedDim(sce, "UMAP")}}{Numeric matrix
#'       \code{[n_cells × n_components]}.}
#'     \item{\code{metadata(sce)$umap}}{List with \code{n_components},
#'       \code{min_dist}, \code{spread}, \code{init}, \code{seed}.}
#'   }
#'
#' @examples
#' \dontrun{
#' sce <- neighbors(sce, n_neighbors = 15)
#' sce <- run_umap(sce, n_components = 2, min_dist = 0.5)
#' dim(reducedDim(sce, "UMAP"))  # ncol(sce) × 2
#' }
#'
#' @seealso \code{\link{neighbors}}, \code{\link{leiden}}
#' @export
run_umap <- function(sce,
                     n_components  = 2L,
                     min_dist      = 0.5,
                     spread        = 1.0,
                     init          = c("random", "spectral"),
                     seed          = 0L,
                     use_neighbors = TRUE) {

    .assert_sce(sce, "run_umap")
    init <- match.arg(init)

    if (init == "spectral") {
        message("run_umap: init='spectral' is NOT deterministic across runs ",
                "(cuML GitHub #6696). Use init='random' for reproducibility.")
    }

    nbrs <- SummarizedExperiment::metadata(sce)$neighbors
    if (is.null(nbrs) || !isTRUE(use_neighbors)) {
        stop("run_umap: metadata(sce)$neighbors not found. Run neighbors(sce) first.")
    }
    if (is.null(nbrs$indices) || is.null(nbrs$distances)) {
        stop("run_umap: metadata(sce)$neighbors must have $indices and $distances.")
    }

    n_components <- as.integer(n_components)
    if (n_components < 1L) stop("run_umap: n_components must be >= 1.")

    idx  <- nbrs$indices
    dist <- nbrs$distances
    if (!is.matrix(idx))  idx  <- as.matrix(idx)
    if (!is.matrix(dist)) dist <- as.matrix(dist)
    storage.mode(idx)  <- "integer"
    storage.mode(dist) <- "double"

    emb <- umap_cpp(idx,
                    dist,
                    n_components,
                    as.double(min_dist),
                    as.double(spread),
                    init,
                    as.integer(seed))

    # emb is NumericMatrix [n_cells × n_components]
    rownames(emb) <- colnames(sce)
    colnames(emb) <- paste0("UMAP", seq_len(n_components))

    SingleCellExperiment::reducedDim(sce, "UMAP") <- emb

    SummarizedExperiment::metadata(sce)$umap <- list(
        n_components = n_components,
        min_dist     = min_dist,
        spread       = spread,
        init         = init,
        seed         = seed
    )

    sce
}
