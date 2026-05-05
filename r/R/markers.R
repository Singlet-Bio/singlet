# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/markers.R
#
# GPU cell-type annotation:
#   score_genes()        — DecoupleR-style gene-set scoring
#   celltypist_predict() — CellTypist reference-based label transfer

#' GPU gene-set scoring
#'
#' Scores each cell against a collection of gene sets using one of four GPU
#' methods (Mlm, Ulm, Wsum, UCell), as implemented in the singlet-gpu
#' \code{marker_score} kernel (DecoupleR API-compatible).
#'
#' @details
#' For each gene set \code{s} in \code{gene_sets}, the activity score for each
#' cell is computed and stored in \code{colData(sce)[[paste0(s, "_", score_name)]]}.
#'
#' Methods:
#' \describe{
#'   \item{\code{"mlm"}}{Multivariate linear model: solves the Tikhonov-regularised
#'     least-squares problem (G^T G + λI)^{-1} G^T x per cell.  Most statistically
#'     rigorous; slowest. (Badia-i-Mompel et al. 2022)}
#'   \item{\code{"ulm"}}{Univariate linear model: per-gene-set Pearson correlation
#'     with the membership vector.  Fast; ignores gene-set overlap.}
#'   \item{\code{"wsum"}}{Weighted sum: Σ_{g ∈ set} w_g · x_{gj}.  Fastest; unit
#'     weights when none are provided.}
#'   \item{\code{"ucell"}}{Rank-based AUC (histogram-approximated, B=4096 bins).
#'     (Andreatta & Carmona 2021).  Non-parametric; robust to outliers.}
#' }
#'
#' \strong{Gene name matching}: gene set member names are matched against
#' \code{rownames(sce)}.  Unmatched genes are silently skipped.  Sets with fewer
#' than 5 matched genes are also skipped (configurable in the C++ kernel; not
#' exposed here for simplicity).
#'
#' @param sce        A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param gene_sets  A named \code{list} of character vectors, where each element
#'   names a gene set and its value lists the gene symbols.
#' @param method     Character.  Scoring method: \code{"mlm"} (default),
#'   \code{"ulm"}, \code{"wsum"}, or \code{"ucell"}.
#' @param score_name Character.  Suffix appended to each set name in colData.
#'   Default: \code{"score"}.
#' @param layer      Character.  Input assay.  Default: \code{"logcounts"}.
#'
#' @return The input \code{sce} with one \code{colData} column per gene set
#'   named \code{<set_name>_<score_name>}.
#'
#' @examples
#' \dontrun{
#' hallmarks <- list(
#'   HYPOXIA     = c("VEGFA", "HIF1A", "LDHA"),
#'   PROLIFERATE = c("MKI67", "TOP2A", "PCNA")
#' )
#' sce <- score_genes(sce, gene_sets = hallmarks, method = "ulm")
#' head(colData(sce)$HYPOXIA_score)
#' }
#'
#' @seealso \code{\link{celltypist_predict}}, \code{\link{rank_genes_groups}}
#' @export
score_genes <- function(sce,
                        gene_sets,
                        method     = c("mlm", "ulm", "wsum", "ucell"),
                        score_name = "score",
                        layer      = "logcounts") {

    .assert_sce(sce, "score_genes")
    method <- match.arg(method)

    if (!is.list(gene_sets) || is.null(names(gene_sets))) {
        stop("score_genes: gene_sets must be a named list of character vectors.")
    }
    if (!layer %in% SummarizedExperiment::assayNames(sce)) {
        stop(sprintf("score_genes: assay '%s' not found.", layer))
    }

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) {
        mat <- methods::as(mat, "dgCMatrix")
    }

    gene_names <- rownames(sce)
    if (is.null(gene_names)) {
        stop("score_genes: rownames(sce) must be set (gene names required for matching).")
    }

    res <- marker_score_cpp(mat,
                            gene_sets,
                            gene_names,
                            method)

    # res$scores is NumericMatrix [n_cells × n_sets]
    # res$set_names is CharacterVector
    scores    <- res$scores
    set_names <- res$set_names

    cd <- SummarizedExperiment::colData(sce)
    for (s_idx in seq_along(set_names)) {
        col_name <- paste0(set_names[s_idx], "_", score_name)
        cd[[col_name]] <- scores[, s_idx]
    }
    SummarizedExperiment::colData(sce) <- cd

    sce
}


#' CellTypist reference-based cell-type prediction
#'
#' Projects cell embeddings onto a pre-trained CellTypist logistic regression
#' model and assigns cell-type labels by argmax probability.
#'
#' @details
#' The function requires a CellTypist model loaded with
#' \code{\link{load_celltypist_model}}.  The model encodes a logistic regression
#' head (weights matrix + intercepts) in device memory.  Inference is a single
#' cuBLAS GEMM followed by per-row softmax and argmax, entirely on GPU.
#'
#' Results are stored in \code{colData(sce)[[key_added]]} as a \code{factor}.
#' The full softmax probability matrix is stored in
#' \code{metadata(sce)$celltypist[[key_added]]$probs}.
#'
#' @param sce         A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param model_path  Character.  Path to a CellTypist \code{.npz} model file.
#'   This is loaded fresh each call (to avoid stale GPU memory across R sessions);
#'   use \code{\link{load_celltypist_model}} to cache the handle.
#' @param use_dimred  Character.  The \code{reducedDim} slot to use as the
#'   embedding (typically \code{"PCA"}).  Default: \code{"PCA"}.
#' @param key_added   Character.  Column name written to \code{colData}.
#'   Default: \code{"celltypist"}.
#'
#' @return The input \code{sce} with \code{colData(sce)[[key_added]]} set to a
#'   factor of predicted cell-type labels.
#'
#' @examples
#' \dontrun{
#' sce <- celltypist_predict(sce,
#'                            model_path = "/path/to/Immune_All_Low.npz",
#'                            use_dimred = "PCA",
#'                            key_added  = "celltypist")
#' table(sce$celltypist)
#' }
#'
#' @seealso \code{\link{load_celltypist_model}}, \code{\link{score_genes}}
#' @export
celltypist_predict <- function(sce,
                                model_path,
                                use_dimred = "PCA",
                                key_added  = "celltypist") {

    .assert_sce(sce, "celltypist_predict")

    available_dimreds <- SingleCellExperiment::reducedDimNames(sce)
    if (!use_dimred %in% available_dimreds) {
        stop(sprintf(
            "celltypist_predict: reducedDim '%s' not found. Available: %s",
            use_dimred,
            paste(available_dimreds, collapse = ", ")))
    }

    model_ptr <- load_celltypist_model_cpp(model_path)

    emb <- SingleCellExperiment::reducedDim(sce, use_dimred)
    if (!is.matrix(emb)) emb <- as.matrix(emb)
    storage.mode(emb) <- "double"

    res <- celltypist_project_cpp(emb, model_ptr)

    # Build factor from 0-based label indices using class_names as levels
    class_names <- res$class_names
    label_factor <- factor(class_names[res$labels + 1L],
                           levels = class_names)

    cd <- SummarizedExperiment::colData(sce)
    cd[[key_added]] <- label_factor
    SummarizedExperiment::colData(sce) <- cd

    # Store probability matrix in metadata (may be large for many classes)
    meta_ct <- SummarizedExperiment::metadata(sce)$celltypist
    if (is.null(meta_ct)) meta_ct <- list()
    meta_ct[[key_added]] <- list(
        probs       = res$probs,
        class_names = class_names,
        use_dimred  = use_dimred,
        model_path  = model_path
    )
    SummarizedExperiment::metadata(sce)$celltypist <- meta_ct

    sce
}


#' Load a CellTypist model into GPU memory
#'
#' Reads a CellTypist \code{.npz} model file, parses the weights and class
#' names, and uploads the weight matrix to GPU device memory.  The returned
#' handle can be passed to \code{\link{celltypist_predict}} to avoid re-loading
#' the model on every call.
#'
#' @param model_path Character.  Path to a CellTypist \code{.npz} file.
#'
#' @return An opaque external pointer (\code{"CelltypistModel"}) wrapping the
#'   device-resident model.  The GPU memory is freed when R garbage-collects
#'   this object.
#'
#' @examples
#' \dontrun{
#' mdl <- load_celltypist_model("/path/to/Immune_All_Low.npz")
#' sce <- celltypist_project_from_handle(sce, mdl, use_dimred = "PCA")
#' }
#'
#' @seealso \code{\link{celltypist_predict}}
#' @export
load_celltypist_model <- function(model_path) {
    if (!file.exists(model_path)) {
        stop(sprintf("load_celltypist_model: file not found: %s", model_path))
    }
    load_celltypist_model_cpp(as.character(model_path))
}
