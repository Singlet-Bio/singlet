# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/disease.R
#
# GPU disease relevance scoring:
#   run_scdrs() — scDRS-style per-cell GWAS disease scoring (cycle 48)
#
# Closest published CPU analog: scDRS Python package (Zhang et al. Nat Genetics 2022).


#' GPU per-cell disease relevance scoring (scDRS)
#'
#' Scores each cell for relevance to GWAS-derived disease gene sets using
#' GPU-accelerated empirical null generation.  Mirrors
#' \href{https://github.com/martinjzhang/scDRS}{scDRS} (Nature Genetics 2022)
#' but generates the 1000 control gene sets and scores entirely on GPU.
#'
#' @details
#' The algorithm:
#' \enumerate{
#'   \item Z-score normalizes gene expression per cell.
#'   \item Computes a raw disease score per (cell, disease) as a weighted sum
#'     of gene z-scores.
#'   \item Samples 1000 control gene sets matched on expression mean and variance.
#'   \item Normalizes the raw score against the empirical null distribution.
#'   \item Computes Monte Carlo p-values and BH FDR per cell.
#' }
#'
#' Results are stored in \code{metadata(sce)$scdrs}:
#' \describe{
#'   \item{\code{raw_score}}{NumericMatrix [n_cells × n_diseases].}
#'   \item{\code{norm_score}}{NumericMatrix [n_cells × n_diseases] (z-scored vs null).}
#'   \item{\code{p_value}}{NumericMatrix [n_cells × n_diseases].}
#'   \item{\code{fdr}}{NumericMatrix [n_cells × n_diseases] (BH-adjusted).}
#'   \item{\code{disease_names}}{CharacterVector.}
#' }
#'
#' @param sce              A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param assay_name       Character.  Default: \code{"logcounts"}.
#' @param disease_gene_sets A named list.  Each element is a named numeric vector
#'   of \code{gene_name → weight} for one disease.  Names of the list become
#'   disease labels.  Weights from MAGMA z-score output.
#' @param n_ctrl_sets      Integer.  Control gene sets per disease.  Default: \code{1000L}.
#' @param seed             Integer.  Default: \code{0L}.
#' @param resource         Character.  Default: \code{"auto"}.
#'
#' @return The input \code{sce} with \code{metadata(sce)$scdrs} populated.
#'
#' @examples
#' \dontrun{
#' # Toy example:
#' gene_sets <- list(
#'   schizophrenia = c(GENE1 = 1.2, GENE2 = 0.8),
#'   t2d           = c(GENE3 = 2.1)
#' )
#' sce <- run_scdrs(sce, disease_gene_sets = gene_sets)
#' }
#'
#' @export
run_scdrs <- function(sce,
                      assay_name       = "logcounts",
                      disease_gene_sets,
                      n_ctrl_sets      = 1000L,
                      seed             = 0L,
                      resource         = "auto") {

    .assert_sce(sce, "run_scdrs")

    if (!assay_name %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_scdrs: assay '%s' not found.", assay_name))
    if (!is.list(disease_gene_sets) || length(disease_gene_sets) == 0)
        stop("run_scdrs: disease_gene_sets must be a non-empty named list.")

    disease_names <- names(disease_gene_sets)
    if (is.null(disease_names) || any(disease_names == ""))
        stop("run_scdrs: all elements of disease_gene_sets must be named.")

    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    gene_names <- rownames(sce)
    if (is.null(gene_names))
        stop("run_scdrs: sce must have rownames (gene names).")

    n_diseases <- length(disease_gene_sets)

    # Build CSR-style flattened gene set tables
    all_indices <- integer(0)
    all_weights <- numeric(0)
    offsets     <- integer(n_diseases + 1L)
    offsets[1]  <- 0L

    for (d in seq_len(n_diseases)) {
        gs   <- disease_gene_sets[[d]]
        gnms <- names(gs)
        if (is.null(gnms)) stop(sprintf("run_scdrs: disease '%s' gene set must be a named vector.", disease_names[d]))

        idx <- match(gnms, gene_names)
        valid <- !is.na(idx)
        if (!any(valid))
            warning(sprintf("run_scdrs: no genes in disease '%s' found in rownames(sce); skipping.", disease_names[d]))

        all_indices <- c(all_indices, idx[valid] - 1L)   # 0-based
        all_weights <- c(all_weights, as.numeric(gs)[valid])
        offsets[d + 1L] <- length(all_indices)
    }

    res <- scdrs_cpp(mat,
                     disease_names,
                     as.integer(all_indices),
                     as.numeric(all_weights),
                     as.integer(offsets),
                     as.integer(n_ctrl_sets),
                     as.integer(seed),
                     as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_scdrs: %s", res$error))

    # Attach disease names as column names to result matrices
    colnames(res$raw_score)  <- disease_names
    colnames(res$norm_score) <- disease_names
    colnames(res$p_value)    <- disease_names
    colnames(res$fdr)        <- disease_names

    SummarizedExperiment::metadata(sce)$scdrs <- list(
        raw_score     = res$raw_score,
        norm_score    = res$norm_score,
        p_value       = res$p_value,
        fdr           = res$fdr,
        disease_names = disease_names,
        n_diseases    = res$n_diseases
    )

    sce
}
