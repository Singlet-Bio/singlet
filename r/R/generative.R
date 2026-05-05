# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/r/R/generative.R
#
# GPU generative models for scRNA-seq:
#   DiscreteDiffusionModel S4 class
#   train_discrete_diffusion() — train phase
#   sample.DiscreteDiffusionModel() — sample(n) surface

# ---------------------------------------------------------------------------
# S4 class: DiscreteDiffusionModel
# ---------------------------------------------------------------------------

#' Discrete diffusion generative model for scRNA-seq counts
#'
#' An S4 class holding a trained discrete diffusion model
#' (device-resident weights) returned by
#' \code{\link{train_discrete_diffusion}}.
#' Use \code{\link[singletGpu]{sample,DiscreteDiffusionModel-method}} to
#' generate synthetic cells.
#'
#' @slot model_ptr  External pointer to the C++/CUDA \code{DiscreteDiffusionModel}.
#' @slot n_genes    Integer.  Number of genes the model was trained on.
#' @slot n_latent   Integer.  Latent dimension.
#' @slot n_steps    Integer.  Number of diffusion steps.
#' @slot n_epochs   Integer.  Training epochs.
#' @slot seed       Integer.  RNG seed used during training.
#'
#' @export
setClass("DiscreteDiffusionModel",
         representation(
             model_ptr = "externalptr",
             n_genes   = "integer",
             n_latent  = "integer",
             n_steps   = "integer",
             n_epochs  = "integer",
             seed      = "integer"
         ))

# ---------------------------------------------------------------------------
# Generic: sample (new generic — mirrors base::sample for model objects)
# ---------------------------------------------------------------------------

#' Sample from a singletGpu generative model
#'
#' Generic for drawing synthetic observations from a fitted generative model.
#'
#' @param x     A fitted generative model (e.g. \code{DiscreteDiffusionModel}).
#' @param n     Integer.  Number of observations to generate.
#' @param ...   Additional method-specific arguments.
#' @return Model-dependent; see individual methods.
#' @export
setGeneric("sample", function(x, n, ...) standardGeneric("sample"))

# ---------------------------------------------------------------------------
# train_discrete_diffusion
# ---------------------------------------------------------------------------

#' Train a discrete diffusion model on scRNA-seq counts
#'
#' Fits a GPU discrete-state diffusion generative model on a count matrix.
#' The model learns the joint distribution of gene counts across cells and can
#' be used to generate synthetic cells via
#' \code{\link[singletGpu]{sample,DiscreteDiffusionModel-method}}.
#'
#' @details
#' The model uses a \code{n_steps}-step forward noising process (Gaussian noise
#' on log-counts projected back to discrete counts) and a reverse denoising
#' network trained via ELBO maximization.  Analog of scDiff / DiffVAE.
#'
#' @param sce       A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param layer     Character.  Input assay (raw counts recommended).
#'   Default: \code{"counts"}.
#' @param n_latent  Integer.  Latent bottleneck dimension.  Default: \code{32L}.
#' @param n_steps   Integer.  Diffusion noise schedule steps.  Default: \code{100L}.
#' @param n_epochs  Integer.  Training epochs.  Default: \code{500L}.
#' @param lr        Numeric.  Adam learning rate.  Default: \code{1e-3}.
#' @param resource  Character.  \code{"auto"} (always GPU).  Default: \code{"auto"}.
#' @param seed      Integer.  RNG seed.  Default: \code{0L}.
#'
#' @return A fitted \code{DiscreteDiffusionModel} S4 object.  Use
#'   \code{sample(model, n)} to draw synthetic cells.
#'
#' @examples
#' \dontrun{
#' model <- train_discrete_diffusion(sce, n_epochs = 200L)
#' synth <- sample(model, 500L)   # 500 synthetic cells as dgCMatrix
#' }
#'
#' @seealso \code{sample,DiscreteDiffusionModel-method}
#' @export
train_discrete_diffusion <- function(sce,
                                     layer    = "counts",
                                     n_latent = 32L,
                                     n_steps  = 100L,
                                     n_epochs = 500L,
                                     lr       = 1e-3,
                                     resource = "auto",
                                     seed     = 0L) {

    .assert_sce(sce, "train_discrete_diffusion")

    if (!layer %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("train_discrete_diffusion: assay '%s' not found.", layer))

    mat <- SummarizedExperiment::assay(sce, layer)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    n_genes  <- as.integer(nrow(sce))
    n_latent <- as.integer(n_latent)
    n_steps  <- as.integer(n_steps)
    n_epochs <- as.integer(n_epochs)
    seed     <- as.integer(seed)
    lr       <- as.double(lr)
    resource <- as.character(resource)

    if (n_latent < 1L) stop("train_discrete_diffusion: n_latent must be >= 1.")
    if (n_steps  < 1L) stop("train_discrete_diffusion: n_steps must be >= 1.")
    if (n_epochs < 1L) stop("train_discrete_diffusion: n_epochs must be >= 1.")

    ptr <- discrete_diffusion_train_cpp(mat, n_latent, n_steps, n_epochs, lr, resource, seed)

    methods::new("DiscreteDiffusionModel",
                 model_ptr = ptr,
                 n_genes   = n_genes,
                 n_latent  = n_latent,
                 n_steps   = n_steps,
                 n_epochs  = n_epochs,
                 seed      = seed)
}

# ---------------------------------------------------------------------------
# sample,DiscreteDiffusionModel-method
# ---------------------------------------------------------------------------

#' Sample synthetic cells from a trained discrete diffusion model
#'
#' Runs the reverse diffusion chain to generate \code{n} synthetic count
#' vectors from the learned distribution.
#'
#' @details
#' The output is a \code{Matrix::dgCMatrix} of shape
#' \code{[n_genes x n]} — the same orientation as singletGpu's sparse
#' matrix convention (genes as rows).  Row names are empty unless the model
#' was built with named genes.
#'
#' @param x     A fitted \code{DiscreteDiffusionModel} from
#'   \code{\link{train_discrete_diffusion}}.
#' @param n     Integer.  Number of synthetic cells to generate.
#' @param seed  Integer.  RNG seed for the reverse chain.  Default: \code{0L}.
#' @param ...   Ignored.
#'
#' @return A \code{Matrix::dgCMatrix} \code{[n_genes x n]}.
#'
#' @examples
#' \dontrun{
#' model <- train_discrete_diffusion(sce)
#' synth <- sample(model, 1000L)
#' dim(synth)   # nrow(sce) x 1000
#' }
#'
#' @seealso \code{\link{train_discrete_diffusion}}
#' @export
setMethod("sample", "DiscreteDiffusionModel",
          function(x, n, seed = 0L, ...) {

    n    <- as.integer(n)
    seed <- as.integer(seed)

    if (n < 1L) stop("sample.DiscreteDiffusionModel: n must be >= 1.")

    res <- discrete_diffusion_sample_cpp(x@model_ptr, n, seed)

    counts <- res$counts  # dgCMatrix [n_genes x n]
    counts
})
