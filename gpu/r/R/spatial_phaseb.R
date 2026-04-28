# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/spatial_phaseb.R
#
# GPU spatial Phase B analysis:
#   run_flash_deconv() — FlashDeconv cell-type deconvolution
#   StagateModel S4 class + fit_stagate() / predict.StagateModel()
#   Cell2fateModel S4 class + fit_cell2fate() / predict.Cell2fateModel()

# ---------------------------------------------------------------------------
# S4 class: StagateModel
# ---------------------------------------------------------------------------

#' STAGATE graph-attention autoencoder model
#'
#' An S4 class holding a fitted STAGATE model (device-resident weights)
#' returned by \code{\link{fit_stagate}}.  Pass to
#' \code{\link[singletGpu]{predict,StagateModel-method}} to obtain latent
#' embeddings for new data.
#'
#' @slot model_ptr  External pointer to the C++/CUDA \code{StagateModel}.
#' @slot n_latent   Integer.  Latent dimension.
#' @slot n_heads    Integer.  Number of attention heads used during training.
#' @slot n_epochs   Integer.  Number of epochs trained.
#' @slot seed       Integer.  RNG seed used during training.
#'
#' @export
setClass("StagateModel",
         representation(
             model_ptr = "externalptr",
             n_latent  = "integer",
             n_heads   = "integer",
             n_epochs  = "integer",
             seed      = "integer"
         ))

# ---------------------------------------------------------------------------
# S4 class: Cell2fateModel
# ---------------------------------------------------------------------------

#' Cell2fate neural ODE model
#'
#' An S4 class holding a fitted Cell2fate model (device-resident weights)
#' returned by \code{\link{fit_cell2fate}}.  Pass to
#' \code{\link[singletGpu]{predict,Cell2fateModel-method}} to obtain fate
#' probabilities and latent time for new data.
#'
#' @slot model_ptr  External pointer to the C++/CUDA \code{Cell2fateModel}.
#' @slot n_latent   Integer.  Latent state dimension.
#' @slot n_epochs   Integer.  Number of epochs trained.
#' @slot seed       Integer.  RNG seed used during training.
#'
#' @export
setClass("Cell2fateModel",
         representation(
             model_ptr = "externalptr",
             n_latent  = "integer",
             n_epochs  = "integer",
             seed      = "integer"
         ))

# ---------------------------------------------------------------------------
# Generic: fit (new generic for NN-style models)
# ---------------------------------------------------------------------------

#' Fit a singletGpu model
#'
#' Generic function to train neural-network model objects returned by
#' \code{\link{fit_stagate}} and \code{\link{fit_cell2fate}}.  Users rarely
#' call this directly — use the named constructor helpers instead.
#'
#' @param object  An unfitted model S4 object.
#' @param ...     Additional arguments passed to the method.
#' @return A fitted model of the same class as \code{object}.
#' @export
setGeneric("fit", function(object, ...) standardGeneric("fit"))

# ---------------------------------------------------------------------------
# run_flash_deconv
# ---------------------------------------------------------------------------

#' GPU spatial cell-type deconvolution (FlashDeconv)
#'
#' Deconvolves the cell-type composition of spatial transcriptomics spots via
#' GPU-accelerated EM, using a single-cell reference.  Analog of RCTD and
#' SPOTlight from Bioconductor.
#'
#' @details
#' For each spot the algorithm estimates a simplex-constrained proportion vector
#' over the \code{n_types} reference cell types.  Convergence is declared per
#' spot when the L1 change in proportions drops below \code{tol}.
#'
#' Proportions are stored as a numeric matrix in
#' \code{colData(sce)$flash_deconv_props} (column per cell type).
#'
#' @param sce         A \code{SingleCellExperiment} with spatial transcriptomics
#'   spots in columns.  Must have a \code{"counts"} assay.
#' @param ref_sce     A \code{SingleCellExperiment} used as the single-cell
#'   reference.  Must share genes (rownames) with \code{sce}.
#' @param ref_labels  Character or factor column in \code{colData(ref_sce)}
#'   giving cell-type labels.  Default: \code{"cell_type"}.
#' @param coords_key  Character.  Column in \code{colData(sce)} giving spatial
#'   X/Y coordinates as a 2-column numeric matrix or two separate columns
#'   \code{coords_key_x} / \code{coords_key_y}.  Default: \code{"spatial"}.
#' @param layer       Character.  Assay to use.  Default: \code{"counts"}.
#' @param n_iter      Integer.  EM iterations.  Default: \code{100L}.
#' @param tol         Numeric.  Per-spot convergence tolerance.  Default: \code{1e-4}.
#' @param seed        Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return The input \code{sce} with:
#'   \describe{
#'     \item{\code{colData(sce)$flash_deconv_props}}{NumericMatrix [spots x types].}
#'     \item{\code{colData(sce)$flash_deconv_converged}}{LogicalVector [spots].}
#'     \item{\code{metadata(sce)$flash_deconv}}{List with \code{n_types},
#'       \code{n_iter_actual}, \code{type_names}.}
#'   }
#'   Mirrors \code{RCTD::run.RCTD()} output convention.
#'
#' @seealso \code{\link{fit_stagate}}, \code{\link{fit_cell2fate}}
#' @export
run_flash_deconv <- function(sce,
                             ref_sce,
                             ref_labels = "cell_type",
                             coords_key = "spatial",
                             layer      = "counts",
                             n_iter     = 100L,
                             tol        = 1e-4,
                             seed       = 0L) {

    .assert_sce(sce,     "run_flash_deconv")
    .assert_sce(ref_sce, "run_flash_deconv (ref_sce)")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_flash_deconv: assay '%s' not found in sce.", layer))
    if (!layer %in% SummarizedExperiment::assayNames(ref_sce))
        stop(sprintf("run_flash_deconv: assay '%s' not found in ref_sce.", layer))

    # Validate shared genes
    shared_genes <- intersect(rownames(sce), rownames(ref_sce))
    if (length(shared_genes) < 10L)
        stop("run_flash_deconv: fewer than 10 shared genes between sce and ref_sce.")

    # Subset to shared genes (preserving sce row order)
    sce_sub     <- sce    [match(shared_genes, rownames(sce)),     ]
    ref_sub     <- ref_sce[match(shared_genes, rownames(ref_sce)), ]

    # Reference cell-type labels → 0-based integers
    if (!ref_labels %in% names(SummarizedExperiment::colData(ref_sce)))
        stop(sprintf("run_flash_deconv: colData column '%s' not found in ref_sce.", ref_labels))
    raw_labels <- SummarizedExperiment::colData(ref_sce)[[ref_labels]]
    if (is.factor(raw_labels)) {
        type_names <- levels(raw_labels)
        label_int  <- as.integer(raw_labels) - 1L
    } else {
        type_names <- sort(unique(as.character(raw_labels)))
        label_int  <- match(as.character(raw_labels), type_names) - 1L
    }
    n_types <- length(type_names)

    # Spatial coordinates: accept a 2-column matrix in colData or two named cols
    cd <- SummarizedExperiment::colData(sce)
    if (coords_key %in% names(cd) && is.matrix(cd[[coords_key]])) {
        coords <- cd[[coords_key]]
        if (ncol(coords) != 2L)
            stop("run_flash_deconv: coords_key matrix must have exactly 2 columns.")
    } else if (paste0(coords_key, "_x") %in% names(cd) &&
               paste0(coords_key, "_y") %in% names(cd)) {
        coords <- cbind(cd[[paste0(coords_key, "_x")]],
                        cd[[paste0(coords_key, "_y")]])
    } else {
        stop(sprintf(
            "run_flash_deconv: '%s' not found as a 2-col matrix or '%s_x'/'%s_y' in colData.",
            coords_key, coords_key, coords_key))
    }
    storage.mode(coords) <- "double"

    # Get matrices
    mat_spot <- SummarizedExperiment::assay(sce_sub, layer)
    mat_ref  <- SummarizedExperiment::assay(ref_sub, layer)
    if (!methods::is(mat_spot, "dgCMatrix")) mat_spot <- methods::as(mat_spot, "dgCMatrix")
    if (!methods::is(mat_ref,  "dgCMatrix")) mat_ref  <- methods::as(mat_ref,  "dgCMatrix")

    n_iter <- as.integer(n_iter)
    seed   <- as.integer(seed)
    tol    <- as.double(tol)

    res <- flash_deconv_cpp(
        mat_spot, mat_ref, as.integer(label_int),
        coords, n_types, n_iter, tol, seed
    )

    # Attach type names to proportions matrix
    props <- res$props
    colnames(props) <- type_names
    rownames(props) <- colnames(sce)

    cd$flash_deconv_props     <- props
    cd$flash_deconv_converged <- res$converged
    SummarizedExperiment::colData(sce) <- cd

    SummarizedExperiment::metadata(sce)$flash_deconv <- list(
        n_types      = n_types,
        n_iter_actual = res$n_iter_actual,
        type_names   = type_names
    )

    sce
}


# ---------------------------------------------------------------------------
# fit_stagate / predict,StagateModel-method
# ---------------------------------------------------------------------------

#' Fit a STAGATE spatial domain detection model
#'
#' Trains a GPU graph-attention autoencoder (STAGATE) on spatial expression
#' data.  The fitted model is returned as a \code{StagateModel} S4 object
#' and can be used with \code{predict()} to obtain latent embeddings.
#'
#' @details
#' STAGATE constructs a spatial neighborhood graph from the coordinates and
#' learns a graph-attention autoencoder that produces latent representations
#' capturing both gene expression and spatial context.
#' Analog of \code{STAGATE.run_STAGATE()} (Python/PyTorch).
#'
#' @param sce         A \code{SingleCellExperiment}.
#' @param coords_key  Character.  Column in \code{colData(sce)} holding spatial
#'   x/y coordinates.  See \code{\link{run_flash_deconv}} for accepted formats.
#'   Default: \code{"spatial"}.
#' @param layer       Character.  Assay to use.  Default: \code{"logcounts"}.
#' @param n_latent    Integer.  Latent dimension.  Default: \code{50L}.
#' @param n_heads     Integer.  Attention heads.  Default: \code{1L}.
#' @param n_epochs    Integer.  Training epochs.  Default: \code{1000L}.
#' @param lr          Numeric.  Adam learning rate.  Default: \code{1e-3}.
#' @param resource    Character.  \code{"auto"} (always GPU).  Default: \code{"auto"}.
#' @param seed        Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return A fitted \code{StagateModel} S4 object.  Use
#'   \code{predict(model, sce)} to embed new data.
#'
#' @seealso \code{\link{run_flash_deconv}}, \code{predict,StagateModel-method}
#' @export
fit_stagate <- function(sce,
                        coords_key = "spatial",
                        layer      = "logcounts",
                        n_latent   = 50L,
                        n_heads    = 1L,
                        n_epochs   = 1000L,
                        lr         = 1e-3,
                        resource   = "auto",
                        seed       = 0L) {

    .assert_sce(sce, "fit_stagate")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("fit_stagate: assay '%s' not found.", layer))

    coords <- .extract_spatial_coords(sce, coords_key, "fit_stagate")

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    n_latent  <- as.integer(n_latent)
    n_heads   <- as.integer(n_heads)
    n_epochs  <- as.integer(n_epochs)
    seed      <- as.integer(seed)
    lr        <- as.double(lr)
    resource  <- as.character(resource)

    ptr <- stagate_fit_cpp(mat, coords, n_latent, n_heads, n_epochs, lr, resource, seed)

    methods::new("StagateModel",
                 model_ptr = ptr,
                 n_latent  = n_latent,
                 n_heads   = n_heads,
                 n_epochs  = n_epochs,
                 seed      = seed)
}

#' Predict STAGATE latent embedding from a fitted model
#'
#' Apply a fitted \code{StagateModel} to a \code{SingleCellExperiment},
#' storing the latent embedding in \code{reducedDim(sce, "STAGATE")}.
#'
#' @param object   A fitted \code{StagateModel} from \code{\link{fit_stagate}}.
#' @param newdata  A \code{SingleCellExperiment}.
#' @param layer    Character.  Assay to use.  Default: \code{"logcounts"}.
#' @param ...      Ignored.
#'
#' @return The input \code{newdata} SCE with
#'   \code{reducedDim(newdata, "STAGATE")} set to the latent embedding
#'   \code{[cells x n_latent]}.
#'
#' @export
setMethod("predict", "StagateModel", function(object, newdata, layer = "logcounts", ...) {

    .assert_sce(newdata, "predict.StagateModel")

    if (!layer %in% SummarizedExperiment::assayNames(newdata))
        stop(sprintf("predict.StagateModel: assay '%s' not found.", layer))

    mat <- SummarizedExperiment::assay(newdata, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    emb <- stagate_predict_cpp(mat, object@model_ptr)

    rownames(emb) <- colnames(newdata)
    colnames(emb) <- paste0("STAGATE_", seq_len(ncol(emb)))

    SingleCellExperiment::reducedDim(newdata, "STAGATE") <- emb
    newdata
})


# ---------------------------------------------------------------------------
# fit_cell2fate / predict,Cell2fateModel-method
# ---------------------------------------------------------------------------

#' Fit a Cell2fate neural ODE model for fate prediction
#'
#' Trains a GPU neural ODE (Cell2fate) on spliced/unspliced count data to
#' learn cell fate trajectories.  The fitted model is returned as a
#' \code{Cell2fateModel} S4 object.
#'
#' @details
#' Cell2fate uses a variational autoencoder with neural ODE dynamics to jointly
#' model RNA velocity and cell fate.  Analog of
#' \code{cell2fate.models.Cell2fate_DynamicalModel.train()} (Python/PyTorch).
#'
#' Requires assays \code{"spliced"} and \code{"unspliced"} in \code{sce}.
#' Use \code{\link{read_pz_velocity_sce}} to populate these from a singlify
#' output directory.
#'
#' @param sce              A \code{SingleCellExperiment} with \code{"spliced"}
#'   and \code{"unspliced"} assays.
#' @param layer_spliced    Character.  Spliced assay name.  Default: \code{"spliced"}.
#' @param layer_unspliced  Character.  Unspliced assay name.  Default: \code{"unspliced"}.
#' @param n_latent         Integer.  Latent state dimension.  Default: \code{20L}.
#' @param n_epochs         Integer.  Training epochs.  Default: \code{500L}.
#' @param lr               Numeric.  Adam learning rate.  Default: \code{1e-3}.
#' @param resource         Character.  \code{"auto"} (always GPU).  Default: \code{"auto"}.
#' @param seed             Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return A fitted \code{Cell2fateModel} S4 object.
#'
#' @seealso \code{predict,Cell2fateModel-method}, \code{\link{read_pz_velocity_sce}}
#' @export
fit_cell2fate <- function(sce,
                          layer_spliced   = "spliced",
                          layer_unspliced = "unspliced",
                          n_latent        = 20L,
                          n_epochs        = 500L,
                          lr              = 1e-3,
                          resource        = "auto",
                          seed            = 0L) {

    .assert_sce(sce, "fit_cell2fate")

    for (lyr in c(layer_spliced, layer_unspliced)) {
        if (!lyr %in% SummarizedExperiment::assayNames(sce))
            stop(sprintf("fit_cell2fate: assay '%s' not found.", lyr))
    }

    mat_s <- SummarizedExperiment::assay(sce, layer_spliced)
    mat_u <- SummarizedExperiment::assay(sce, layer_unspliced)
    if (!methods::is(mat_s, "dgCMatrix")) mat_s <- methods::as(mat_s, "dgCMatrix")
    if (!methods::is(mat_u, "dgCMatrix")) mat_u <- methods::as(mat_u, "dgCMatrix")

    n_latent <- as.integer(n_latent)
    n_epochs <- as.integer(n_epochs)
    seed     <- as.integer(seed)
    lr       <- as.double(lr)
    resource <- as.character(resource)

    ptr <- cell2fate_fit_cpp(mat_s, mat_u, n_latent, n_epochs, lr, resource, seed)

    methods::new("Cell2fateModel",
                 model_ptr = ptr,
                 n_latent  = n_latent,
                 n_epochs  = n_epochs,
                 seed      = seed)
}

#' Predict cell fate probabilities from a fitted Cell2fate model
#'
#' Apply a fitted \code{Cell2fateModel} to produce fate state probabilities
#' and latent pseudotime.
#'
#' @param object           A fitted \code{Cell2fateModel}.
#' @param newdata          A \code{SingleCellExperiment} with spliced/unspliced assays.
#' @param n_states         Integer.  Number of fate states.  Default: \code{5L}.
#' @param layer_spliced    Character.  Default: \code{"spliced"}.
#' @param layer_unspliced  Character.  Default: \code{"unspliced"}.
#' @param ...              Ignored.
#'
#' @return The input \code{newdata} SCE with:
#'   \describe{
#'     \item{\code{colData(newdata)$cell2fate_latent_time}}{Pseudotime vector.}
#'     \item{\code{reducedDim(newdata, "Cell2fate")}}{Fate probability matrix
#'       \code{[cells x n_states]}.}
#'   }
#'
#' @export
setMethod("predict", "Cell2fateModel",
          function(object, newdata, n_states = 5L,
                   layer_spliced = "spliced", layer_unspliced = "unspliced", ...) {

    .assert_sce(newdata, "predict.Cell2fateModel")

    for (lyr in c(layer_spliced, layer_unspliced)) {
        if (!lyr %in% SummarizedExperiment::assayNames(newdata))
            stop(sprintf("predict.Cell2fateModel: assay '%s' not found.", lyr))
    }

    mat_s <- SummarizedExperiment::assay(newdata, layer_spliced)
    mat_u <- SummarizedExperiment::assay(newdata, layer_unspliced)
    if (!methods::is(mat_s, "dgCMatrix")) mat_s <- methods::as(mat_s, "dgCMatrix")
    if (!methods::is(mat_u, "dgCMatrix")) mat_u <- methods::as(mat_u, "dgCMatrix")

    n_states <- as.integer(n_states)

    res <- cell2fate_predict_cpp(mat_s, mat_u, object@model_ptr, n_states)

    fp <- res$fate_probs
    rownames(fp) <- colnames(newdata)
    colnames(fp) <- res$state_names

    SingleCellExperiment::reducedDim(newdata, "Cell2fate") <- fp

    cd <- SummarizedExperiment::colData(newdata)
    cd$cell2fate_latent_time <- res$latent_time
    SummarizedExperiment::colData(newdata) <- cd

    newdata
})


# ---------------------------------------------------------------------------
# Internal helper: extract spatial coordinates from colData
# ---------------------------------------------------------------------------
.extract_spatial_coords <- function(sce, coords_key, caller) {
    cd <- SummarizedExperiment::colData(sce)
    if (coords_key %in% names(cd) && is.matrix(cd[[coords_key]])) {
        m <- cd[[coords_key]]
        if (ncol(m) != 2L)
            stop(sprintf("%s: coords_key matrix must have exactly 2 columns.", caller))
        storage.mode(m) <- "double"
        return(m)
    }
    xk <- paste0(coords_key, "_x")
    yk <- paste0(coords_key, "_y")
    if (xk %in% names(cd) && yk %in% names(cd)) {
        m <- cbind(as.double(cd[[xk]]), as.double(cd[[yk]]))
        return(m)
    }
    stop(sprintf(
        "%s: coords_key '%s' not found as 2-col matrix or as '%s_x'/'%s_y' in colData.",
        caller, coords_key, coords_key, coords_key))
}
