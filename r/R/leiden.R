# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/leiden.R
#
# GPU Leiden community detection.
#   leiden()  — run Leiden clustering from the neighbors graph

#' GPU Leiden community detection
#'
#' Runs the Leiden algorithm on the kNN graph stored in
#' \code{metadata(sce)$neighbors} (built by \code{\link{neighbors}}), using the
#' singlet-gpu cuGraph Leiden adapter.
#'
#' @details
#' Cluster labels are stored in \code{colData(sce)[[column_name]]} as a
#' \code{factor}.
#'
#' Edge weights are derived from kNN distances via \code{weight_function}:
#' \describe{
#'   \item{\code{"connectivity"}}{w = 1/(1+d²), range (0,1]. scanpy default.}
#'   \item{\code{"inverse"}}{w = 1/(1+d), gentler distance falloff.}
#'   \item{\code{"gaussian"}}{w = exp(-d²/2σ²), σ computed as median(d) on device.}
#' }
#'
#' \strong{Prerequisite}: \code{metadata(sce)$neighbors} must exist (call
#' \code{neighbors(sce)} first).
#'
#' \strong{Determinism}: Leiden is deterministic given a fixed \code{seed}
#' and a sorted kNN input (cuGraph sorts internally).
#'
#' @param sce             A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param resolution      Numeric.  Modularity resolution parameter.  Higher
#'   values produce more, smaller clusters.  Default: \code{1.0}.
#' @param seed            Integer.  RNG seed for cuGraph Leiden.  Default: \code{0}.
#' @param weight_function Character.  Edge-weight construction:
#'   \code{"connectivity"} (default), \code{"inverse"}, or \code{"gaussian"}.
#' @param column_name     Character.  Name of the \code{colData} column to store
#'   cluster labels.  Default: \code{"leiden"}.
#'
#' @return The input \code{sce} with \code{colData(sce)[[column_name]]} set to a
#'   \code{factor} of cluster labels, and \code{metadata(sce)$leiden} recording
#'   provenance (n_clusters, modularity, iterations, resolution, seed).
#'
#' @examples
#' \dontrun{
#' sce <- neighbors(sce, n_neighbors = 15)
#' sce <- leiden(sce, resolution = 1.0)
#' table(sce$leiden)
#' }
#'
#' @seealso \code{\link{neighbors}}, \code{\link{run_umap}}
#' @export
leiden <- function(sce,
                   resolution      = 1.0,
                   seed            = 0L,
                   weight_function = c("connectivity", "inverse", "gaussian"),
                   column_name     = "leiden") {

    .assert_sce(sce, "leiden")
    weight_function <- match.arg(weight_function)

    nbrs <- SummarizedExperiment::metadata(sce)$neighbors
    if (is.null(nbrs)) {
        stop("leiden: metadata(sce)$neighbors not found. Run neighbors(sce) first.")
    }
    if (is.null(nbrs$indices) || is.null(nbrs$distances)) {
        stop("leiden: metadata(sce)$neighbors must have $indices and $distances. Re-run neighbors(sce).")
    }

    idx  <- nbrs$indices
    dist <- nbrs$distances
    if (!is.matrix(idx))  idx  <- as.matrix(idx)
    if (!is.matrix(dist)) dist <- as.matrix(dist)
    storage.mode(idx)  <- "integer"
    storage.mode(dist) <- "double"

    res <- leiden_cpp(idx,
                      dist,
                      as.double(resolution),
                      as.integer(seed),
                      weight_function)

    # Labels are 0-based integers; convert to factor with 1-based level labels
    labels_int <- res$labels + 1L  # 1-based for R factor convention
    lbl_factor <- factor(labels_int,
                         levels  = seq_len(res$n_clusters),
                         labels  = as.character(seq_len(res$n_clusters)))

    cd <- SummarizedExperiment::colData(sce)
    cd[[column_name]] <- lbl_factor
    SummarizedExperiment::colData(sce) <- cd

    SummarizedExperiment::metadata(sce)$leiden <- list(
        n_clusters      = res$n_clusters,
        modularity      = res$modularity,
        iterations      = res$iterations,
        resolution      = resolution,
        weight_function = weight_function,
        seed            = seed,
        column_name     = column_name
    )

    sce
}
