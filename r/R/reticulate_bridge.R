# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/reticulate_bridge.R
#
# Optional reticulate bridge to rapids-singlecell Python functions.
#
# These functions are OPT-IN — they require:
#   1. The `reticulate` R package.
#   2. The `singlet_gpu` Python wheel installed in the active Python env.
#      (pip install git+https://github.com/zdebruine/Singlet-AI#subdirectory=singlet-gpu/python)
#
# If either dependency is absent the functions throw a clear error message.
# The native Rcpp path (load_pz.R, preprocess.R, reduce.R) does NOT depend
# on reticulate and is always available.

#' Harmony batch correction via rapids-singlecell (reticulate bridge)
#'
#' Calls \code{rapids_singlecell.pp.harmony_integrate} on the PCA embedding
#' stored in \code{reducedDim(sce, "PCA")}, then stores the corrected embedding
#' in \code{reducedDim(sce, "X_pca_harmony")}.
#'
#' @details
#' **Requires** \code{reticulate} and the \code{singlet_gpu} Python wheel.
#' If either is absent an informative error is thrown.
#'
#' The SCE is marshalled to AnnData via \code{reticulate} before the Python call
#' and the corrected embedding is pulled back into the SCE on return.  This
#' incurs an extra host↔Python memory copy but avoids any R-side re-implementation
#' of Harmony.
#'
#' @param sce       A \code{\link[SingleCellExperiment]{SingleCellExperiment}}
#'   with \code{reducedDim(sce, "PCA")} populated (from \code{\link{run_pca}}).
#' @param batch_key Character scalar.  Column name in \code{colData(sce)} that
#'   identifies the batch variable (e.g. \code{"sample_id"}).
#' @param key       Character scalar.  Name of the corrected embedding stored
#'   in \code{reducedDim}.  Default: \code{"X_pca_harmony"}.
#' @param ...       Additional keyword arguments forwarded to
#'   \code{rapids_singlecell.pp.harmony_integrate} (Python).
#'
#' @return The input \code{sce} with \code{reducedDim(sce, key)} populated.
#'
#' @examples
#' \dontrun{
#' # Requires reticulate + singlet_gpu Python wheel
#' sce <- rapids_harmony(sce, batch_key = "sample_id")
#' dim(reducedDim(sce, "X_pca_harmony"))
#' }
#'
#' @export
rapids_harmony <- function(sce, batch_key, key = "X_pca_harmony", ...) {
    .check_reticulate()
    rsc <- .import_singlet_gpu_python()

    .assert_sce(sce, "rapids_harmony")

    if (!"PCA" %in% SingleCellExperiment::reducedDimNames(sce)) {
        stop("rapids_harmony: reducedDim 'PCA' not found. Run run_pca() first.")
    }
    if (!batch_key %in% names(SummarizedExperiment::colData(sce))) {
        stop(sprintf(
            "rapids_harmony: batch_key '%s' not found in colData(sce).",
            batch_key))
    }

    # Marshal SCE to AnnData via reticulate
    adata <- .sce_to_anndata(sce)

    # Call rapids-singlecell harmony
    rsc$pp$harmony_integrate(adata,
                             key    = batch_key,
                             basis  = "X_pca",
                             adjusted_basis = "X_pca_harmony",
                             ...)

    # Pull corrected embedding back into SCE
    harmony_embed <- reticulate::py_to_r(
        adata$obsm["X_pca_harmony"])
    rownames(harmony_embed) <- colnames(sce)

    SingleCellExperiment::reducedDim(sce, key) <- harmony_embed
    sce
}


#' UMAP embedding via rapids-singlecell (reticulate bridge)
#'
#' Calls \code{rapids_singlecell.tl.umap} on the specified basis embedding,
#' then stores the 2-D UMAP coordinates in \code{reducedDim(sce, "UMAP")}.
#'
#' @details
#' **Requires** \code{reticulate} and the \code{singlet_gpu} Python wheel.
#'
#' @param sce   A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param basis Character scalar.  Name of the reducedDim to use as input
#'   (e.g. \code{"X_pca_harmony"} or \code{"PCA"}).  Default: \code{"PCA"}.
#' @param key   Character scalar.  Name of the output reducedDim.
#'   Default: \code{"UMAP"}.
#' @param n_neighbors Integer.  k-NN graph size.  Default: \code{15L}.
#' @param min_dist    Numeric.  UMAP min_dist parameter.  Default: \code{0.5}.
#' @param ...   Additional keyword arguments forwarded to
#'   \code{rapids_singlecell.tl.umap} (Python).
#'
#' @return The input \code{sce} with \code{reducedDim(sce, key)} populated.
#'
#' @examples
#' \dontrun{
#' sce <- rapids_umap(sce, basis = "X_pca_harmony")
#' dim(reducedDim(sce, "UMAP"))  # ncol(sce) × 2
#' }
#'
#' @export
rapids_umap <- function(sce,
                        basis       = "PCA",
                        key         = "UMAP",
                        n_neighbors = 15L,
                        min_dist    = 0.5,
                        ...) {
    .check_reticulate()
    rsc <- .import_singlet_gpu_python()

    .assert_sce(sce, "rapids_umap")

    if (!basis %in% SingleCellExperiment::reducedDimNames(sce)) {
        stop(sprintf(
            "rapids_umap: reducedDim '%s' not found. Run run_pca() or ",
            basis), "rapids_harmony() first.")
    }

    adata <- .sce_to_anndata(sce)

    # Map SCE reducedDim name to AnnData obsm key
    obsm_key <- if (basis == "PCA") "X_pca" else basis

    rsc$pp$neighbors(adata,
                     n_neighbors = as.integer(n_neighbors),
                     use_rep     = obsm_key)
    rsc$tl$umap(adata, min_dist = min_dist, ...)

    umap_embed <- reticulate::py_to_r(adata$obsm["X_umap"])
    rownames(umap_embed) <- colnames(sce)
    colnames(umap_embed) <- c("UMAP_1", "UMAP_2")

    SingleCellExperiment::reducedDim(sce, key) <- umap_embed
    sce
}
