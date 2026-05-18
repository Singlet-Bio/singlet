# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/velocity.R
#
# GPU RNA velocity:
#   velocity_moments()  — compute first-order moments (Ms, Mu)
#   velocity_prep()     — estimate gamma + compute velocity matrix
#   read_pz_velocity_sce() — load spliced/unspliced from singlet output dir
#
# Analog of velociraptor::scvelo() / scVelo Python API.

#' Load spliced and unspliced assays from a singlet output directory
#'
#' Reads \code{exon_counts.1pz} (spliced) and \code{intron_counts.1pz}
#' (unspliced) from a singlet pipeline output directory and returns a
#' \code{SingleCellExperiment} with assays \code{"spliced"} and
#' \code{"unspliced"}.
#'
#' @details
#' This helper is provided because \code{velocity_moments()} requires an SCE
#' with \code{assays(sce)$spliced} and \code{assays(sce)$unspliced} — the
#' singlet-native names for the exon and intron count matrices respectively.
#'
#' If an existing SCE is provided via \code{sce}, the two assays are added to
#' it (overwriting if already present).  Otherwise a new SCE is created from
#' the two matrices.
#'
#' @param pz_dir Character.  Path to a singlet output directory containing
#'   \code{exon_counts.1pz} and \code{intron_counts.1pz}.
#' @param sce    Optional \code{SingleCellExperiment}.  If provided, the
#'   spliced/unspliced assays are added to it.  Default: \code{NULL} (new SCE).
#'
#' @return A \code{SingleCellExperiment} with assays \code{"spliced"} and
#'   \code{"unspliced"}.
#'
#' @seealso \code{\link{velocity_moments}}, \code{\link{velocity_prep}}
#' @export
read_pz_velocity_sce <- function(pz_dir, sce = NULL) {
    spliced_path   <- file.path(pz_dir, "exon_counts.1pz")
    unspliced_path <- file.path(pz_dir, "intron_counts.1pz")

    if (!file.exists(spliced_path))
        stop(sprintf("read_pz_velocity_sce: 'exon_counts.1pz' not found in %s", pz_dir))
    if (!file.exists(unspliced_path))
        stop(sprintf("read_pz_velocity_sce: 'intron_counts.1pz' not found in %s", pz_dir))

    # Use the singletGpu loader to read each .1pz as a sparse matrix
    s_data  <- load_pz_to_sce_cpp(pz_dir, "scrna_spliced",   FALSE)
    u_data  <- load_pz_to_sce_cpp(pz_dir, "scrna_unspliced", FALSE)

    if (is.null(sce)) {
        sce <- SingleCellExperiment::SingleCellExperiment(
            assays = list(spliced = s_data$counts, unspliced = u_data$counts)
        )
    } else {
        .assert_sce(sce, "read_pz_velocity_sce")
        SummarizedExperiment::assay(sce, "spliced")   <- s_data$counts
        SummarizedExperiment::assay(sce, "unspliced") <- u_data$counts
    }

    sce
}


#' GPU RNA-velocity first-order moments
#'
#' Computes kNN-smoothed first-order moments of the spliced (\code{Ms}) and
#' unspliced (\code{Mu}) count matrices.  Analog of
#' \code{velociraptor::scvelo(mode = "moments")}.
#'
#' @details
#' For each cell \eqn{i}, the moment is computed as the mean expression across
#' the \eqn{k} nearest neighbours (including the cell itself):
#' \deqn{M^{(s)}_i = \frac{1}{k} \sum_{j \in \text{KNN}(i)} S_{gj}}
#' The computation is entirely on device; no per-gene host traffic.
#'
#' Results are stored in \code{assays(sce)$Ms} and \code{assays(sce)$Mu}.
#'
#' @param sce              A \code{\link[SingleCellExperiment]{SingleCellExperiment}}
#'   with assays \code{"spliced"} and \code{"unspliced"}.  Use
#'   \code{\link{read_pz_velocity_sce}} to populate these from a singlet
#'   output directory.
#' @param n_neighbors      Integer.  k for kNN smoothing.  Default: \code{30L}.
#' @param use_dimred       Character.  \code{reducedDim} slot used to build
#'   the kNN graph (must already be computed).  Default: \code{"PCA"}.
#' @param layer_spliced    Character.  Name of spliced count assay.
#'   Default: \code{"spliced"}.
#' @param layer_unspliced  Character.  Name of unspliced count assay.
#'   Default: \code{"unspliced"}.
#'
#' @return The input \code{sce} with \code{assays(sce)$Ms} and
#'   \code{assays(sce)$Mu} set.
#'
#' @examples
#' \dontrun{
#' sce <- read_pz_velocity_sce("path/to/GSM4037629/")
#' sce <- run_pca(sce)
#' sce <- build_neighbors(sce)
#' sce <- velocity_moments(sce)
#' }
#'
#' @seealso \code{\link{velocity_prep}}, \code{\link{read_pz_velocity_sce}}
#' @export
velocity_moments <- function(sce,
                             n_neighbors     = 30L,
                             use_dimred      = "PCA",
                             layer_spliced   = "spliced",
                             layer_unspliced = "unspliced") {

    .assert_sce(sce, "velocity_moments")

    for (lyr in c(layer_spliced, layer_unspliced)) {
        if (!lyr %in% SummarizedExperiment::assayNames(sce))
            stop(sprintf("velocity_moments: assay '%s' not found.", lyr))
    }
    if (!use_dimred %in% SingleCellExperiment::reducedDimNames(sce))
        stop(sprintf("velocity_moments: reducedDim '%s' not found.", use_dimred))

    n_neighbors <- as.integer(n_neighbors)

    mat_s <- SummarizedExperiment::assay(sce, layer_spliced)
    mat_u <- SummarizedExperiment::assay(sce, layer_unspliced)

    if (!methods::is(mat_s, "dgCMatrix")) mat_s <- methods::as(mat_s, "dgCMatrix")
    if (!methods::is(mat_u, "dgCMatrix")) mat_u <- methods::as(mat_u, "dgCMatrix")

    # Retrieve kNN indices — check metadata$neighbors first, then compute inline
    neighbors_meta <- SummarizedExperiment::metadata(sce)$neighbors
    if (!is.null(neighbors_meta) && !is.null(neighbors_meta$indices)) {
        knn_idx <- neighbors_meta$indices
    } else {
        # Compute kNN on the fly if not already stored
        emb <- SingleCellExperiment::reducedDim(sce, use_dimred)
        if (!is.matrix(emb)) emb <- as.matrix(emb)
        storage.mode(emb) <- "double"
        knn_res <- neighbors_cpp(emb, n_neighbors, "euclidean", 0L)
        knn_idx <- knn_res$indices
        SummarizedExperiment::metadata(sce)$neighbors <- knn_res
    }

    # Truncate knn_idx to n_neighbors columns if necessary
    if (ncol(knn_idx) > n_neighbors) {
        knn_idx <- knn_idx[, seq_len(n_neighbors), drop = FALSE]
    }

    res <- velocity_moments_cpp(mat_s, mat_u, knn_idx, n_neighbors)

    SummarizedExperiment::assay(sce, "Ms") <- res$Ms
    SummarizedExperiment::assay(sce, "Mu") <- res$Mu

    sce
}


#' GPU RNA velocity estimation
#'
#' Estimates per-gene velocity parameters (gamma, offset) from the smoothed
#' first-order moments (\code{Ms}, \code{Mu}), optionally computing the
#' velocity matrix.  Analog of \code{velociraptor::scvelo(mode="steady_state")}.
#'
#' @details
#' For each gene, gamma is estimated by minimising the residual of the
#' steady-state equation \eqn{U \approx \gamma S} using on-device least-squares
#' regression.  Genes are then filtered by minimum spliced/unspliced counts
#' and a likelihood quantile filter.
#'
#' When \code{compute_velocity = TRUE}, the velocity matrix
#' \eqn{V = S - \hat{\gamma} U} is computed on device and stored in
#' \code{assays(sce)$velocity}.
#'
#' Per-gene parameters are stored in \code{rowData(sce)$velocity_gamma} and
#' \code{rowData(sce)$velocity_filter}.
#'
#' @param sce               A \code{SingleCellExperiment} with assays \code{"Ms"}
#'   and \code{"Mu"} (from \code{\link{velocity_moments}}).
#' @param mode              Character.  Currently only \code{"steady_state"}.
#' @param min_S_count       Integer.  Minimum total spliced reads per gene.
#'   Default: \code{10L}.
#' @param min_U_count       Integer.  Minimum total unspliced reads per gene.
#'   Default: \code{5L}.
#' @param top_n_quantile    Integer.  Retain top N% of genes by likelihood.
#'   Default: \code{5L}.
#' @param compute_velocity  Logical.  Compute the full velocity matrix.
#'   Default: \code{TRUE}.
#'
#' @return The input \code{sce} with \code{rowData(sce)$velocity_gamma},
#'   \code{rowData(sce)$velocity_filter}, and (if
#'   \code{compute_velocity=TRUE}) \code{assays(sce)$velocity} set.
#'
#' @examples
#' \dontrun{
#' sce <- velocity_moments(sce)
#' sce <- velocity_prep(sce, compute_velocity = TRUE)
#' }
#'
#' @seealso \code{\link{velocity_moments}}, \code{\link{read_pz_velocity_sce}}
#' @export
velocity_prep <- function(sce,
                          mode             = "steady_state",
                          min_S_count      = 10L,
                          min_U_count      = 5L,
                          top_n_quantile   = 5L,
                          compute_velocity = TRUE) {

    .assert_sce(sce, "velocity_prep")

    if (mode != "steady_state")
        stop("velocity_prep: only mode='steady_state' is currently supported.")

    for (lyr in c("Ms", "Mu")) {
        if (!lyr %in% SummarizedExperiment::assayNames(sce))
            stop(sprintf("velocity_prep: assay '%s' not found. Run velocity_moments() first.", lyr))
    }

    Ms <- SummarizedExperiment::assay(sce, "Ms")
    Mu <- SummarizedExperiment::assay(sce, "Mu")

    if (!is.matrix(Ms)) Ms <- as.matrix(Ms)
    if (!is.matrix(Mu)) Mu <- as.matrix(Mu)
    storage.mode(Ms) <- "double"
    storage.mode(Mu) <- "double"

    res <- velocity_prep_compute_cpp(
        Ms, Mu,
        as.integer(min_S_count),
        as.integer(min_U_count),
        as.integer(top_n_quantile),
        as.logical(compute_velocity)
    )

    rd <- SummarizedExperiment::rowData(sce)
    rd$velocity_gamma  <- res$velocity_gamma
    rd$velocity_filter <- res$velocity_filter
    SummarizedExperiment::rowData(sce) <- rd

    if (compute_velocity && !is.null(res$velocity)) {
        SummarizedExperiment::assay(sce, "velocity") <- res$velocity
    }

    sce
}
