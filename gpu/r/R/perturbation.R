# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/perturbation.R
#
# GPU perturbation-response modeling:
#   PerturbGraphModel S4 class
#   fit_perturb_graph() — training phase
#   predict,PerturbGraphModel-method — in-silico perturbation prediction

# ---------------------------------------------------------------------------
# S4 class: PerturbGraphModel
# ---------------------------------------------------------------------------

#' PerturbGraph neural perturbation response model
#'
#' An S4 class holding a fitted PerturbGraph model (device-resident weights)
#' returned by \code{\link{fit_perturb_graph}}.
#' Use \code{\link[singletGpu]{predict,PerturbGraphModel-method}} to predict
#' transcriptional responses to in-silico perturbations.
#'
#' @slot model_ptr   External pointer to the C++/CUDA \code{PerturbGraphModel}.
#' @slot n_perturb   Integer.  Number of perturbation conditions.
#' @slot n_latent    Integer.  Encoder latent dimension.
#' @slot n_epochs    Integer.  Training epochs.
#' @slot seed        Integer.  RNG seed used during training.
#' @slot perturb_levels  Character vector of perturbation names (1-indexed, level
#'   0 = control).
#'
#' @export
setClass("PerturbGraphModel",
         representation(
             model_ptr      = "externalptr",
             n_perturb      = "integer",
             n_latent       = "integer",
             n_epochs       = "integer",
             seed           = "integer",
             perturb_levels = "character"
         ))

# ---------------------------------------------------------------------------
# fit_perturb_graph
# ---------------------------------------------------------------------------

#' Fit a PerturbGraph model for in-silico perturbation prediction
#'
#' Trains a GPU neural perturbation graph on paired control + perturbed
#' expression data.  Analog of scGen or GEARS.
#'
#' @details
#' The model learns a latent perturbation-specific offset such that for a
#' control cell embedding \eqn{z}, the predicted response to perturbation
#' \eqn{k} is \eqn{z + \delta_k}.  The decoder maps back to gene expression
#' space.  Analog of \code{scGen.train()} / \code{GEARS.train()}.
#'
#' @param sce            A \code{SingleCellExperiment} containing both control
#'   and perturbed cells.
#' @param perturb_key    Character.  Column in \code{colData(sce)} giving
#'   perturbation condition labels.  Control cells must have a distinct label
#'   (use \code{control_label} to specify which one).
#' @param control_label  Character.  Value in \code{colData(sce)[[perturb_key]]}
#'   indicating control cells.  Default: \code{"control"}.
#' @param layer          Character.  Input assay.  Default: \code{"logcounts"}.
#' @param n_latent       Integer.  Encoder latent dimension.  Default: \code{64L}.
#' @param n_epochs       Integer.  Training epochs.  Default: \code{400L}.
#' @param lr             Numeric.  Adam learning rate.  Default: \code{1e-3}.
#' @param resource       Character.  \code{"auto"} (always GPU).  Default: \code{"auto"}.
#' @param seed           Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return A fitted \code{PerturbGraphModel} S4 object.
#'
#' @examples
#' \dontrun{
#' # colData(sce)$condition: "control", "gene_A_KD", "gene_B_OE", ...
#' model <- fit_perturb_graph(sce, perturb_key = "condition")
#' pred <- predict(model, sce_ctrl, target_perturbations = c("gene_A_KD"))
#' }
#'
#' @seealso \code{predict,PerturbGraphModel-method}
#' @export
fit_perturb_graph <- function(sce,
                              perturb_key    = "perturbation",
                              control_label  = "control",
                              layer          = "logcounts",
                              n_latent       = 64L,
                              n_epochs       = 400L,
                              lr             = 1e-3,
                              resource       = "auto",
                              seed           = 0L) {

    .assert_sce(sce, "fit_perturb_graph")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("fit_perturb_graph: assay '%s' not found.", layer))
    if (!perturb_key %in% names(SummarizedExperiment::colData(sce)))
        stop(sprintf("fit_perturb_graph: colData column '%s' not found.", perturb_key))

    raw_labels <- SummarizedExperiment::colData(sce)[[perturb_key]]
    raw_labels <- as.character(raw_labels)

    if (!control_label %in% raw_labels)
        stop(sprintf(
            "fit_perturb_graph: control_label '%s' not found in colData[[%s]].",
            control_label, perturb_key))

    # Encode: control = 0, perturbations = 1..K
    all_levels <- c(control_label, sort(setdiff(unique(raw_labels), control_label)))
    label_int  <- match(raw_labels, all_levels) - 1L   # 0-based

    n_perturb <- length(all_levels) - 1L   # number of non-control conditions

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    n_latent <- as.integer(n_latent)
    n_epochs <- as.integer(n_epochs)
    seed     <- as.integer(seed)
    lr       <- as.double(lr)
    resource <- as.character(resource)

    ptr <- perturb_graph_fit_cpp(mat, as.integer(label_int),
                                 n_latent, n_epochs, lr, resource, seed)

    methods::new("PerturbGraphModel",
                 model_ptr      = ptr,
                 n_perturb      = as.integer(n_perturb),
                 n_latent       = n_latent,
                 n_epochs       = n_epochs,
                 seed           = seed,
                 perturb_levels = all_levels)
}

# ---------------------------------------------------------------------------
# predict,PerturbGraphModel-method
# ---------------------------------------------------------------------------

#' Predict transcriptional responses to in-silico perturbations
#'
#' Apply a fitted \code{PerturbGraphModel} to control cells to predict how
#' each cell would respond to the specified perturbation(s).
#'
#' @details
#' For each target perturbation, the predicted expression matrix and the delta
#' (predicted minus control) are returned.  Results are stored in
#' \code{metadata(sce)$perturb_predictions} as a named list.
#'
#' @param object               A fitted \code{PerturbGraphModel}.
#' @param newdata              A \code{SingleCellExperiment} of control cells.
#' @param target_perturbations Character vector.  Names of perturbation
#'   conditions to simulate (must match names seen during training).
#'   Default: all non-control conditions.
#' @param layer                Character.  Assay to use.  Default: \code{"logcounts"}.
#' @param ...                  Ignored.
#'
#' @return The input \code{newdata} with
#'   \code{metadata(newdata)$perturb_predictions} set to a named list.
#'   Each element is itself a list with:
#'   \describe{
#'     \item{\code{predicted}}{NumericMatrix [cells x genes] — predicted expression.}
#'     \item{\code{delta}}{NumericMatrix [cells x genes] — log-fold change vs control.}
#'   }
#'
#' @seealso \code{\link{fit_perturb_graph}}
#' @export
setMethod("predict", "PerturbGraphModel",
          function(object, newdata, target_perturbations = NULL, layer = "logcounts", ...) {

    .assert_sce(newdata, "predict.PerturbGraphModel")

    if (!layer %in% SummarizedExperiment::assayNames(newdata))
        stop(sprintf("predict.PerturbGraphModel: assay '%s' not found.", layer))

    # Default: all non-control perturbations
    if (is.null(target_perturbations)) {
        target_perturbations <- object@perturb_levels[-1L]
    }

    # Map names → 1-based integer IDs (index in perturb_levels; 0 = control)
    target_ids <- match(target_perturbations, object@perturb_levels)
    if (any(is.na(target_ids)))
        stop(sprintf(
            "predict.PerturbGraphModel: unknown perturbation(s): %s",
            paste(target_perturbations[is.na(target_ids)], collapse = ", ")))
    # perturb_graph_predict_cpp expects 1-based IDs (control = 0 at index 1 → skip)
    target_ids_0based <- target_ids - 1L   # control is 0, first pert is 1

    mat <- SummarizedExperiment::assay(newdata, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    res <- perturb_graph_predict_cpp(mat, object@model_ptr, as.integer(target_ids_0based))

    # res$predicted and res$delta are [cells × genes] for each target
    # Since we may have multiple targets, the kernel returns one combined block
    # (repeated: target_ids_0based is a vector).
    # Wrap into named list by perturbation.
    n_targets <- length(target_perturbations)
    preds <- vector("list", n_targets)
    names(preds) <- target_perturbations

    n_cells <- ncol(newdata)
    n_genes <- nrow(newdata)

    for (i in seq_len(n_targets)) {
        # The kernel returns predictions for all targets stacked; each block is
        # n_cells rows.  Slice accordingly.
        row_start <- (i - 1L) * n_cells + 1L
        row_end   <- i * n_cells
        preds[[i]] <- list(
            predicted = res$predicted[row_start:row_end, , drop = FALSE],
            delta     = res$delta    [row_start:row_end, , drop = FALSE]
        )
    }

    SummarizedExperiment::metadata(newdata)$perturb_predictions <- preds
    newdata
})
