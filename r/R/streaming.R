# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/streaming.R
#
# Streaming pipeline over multiple .1pz files.
#   run_pipeline()  — lognorm + HVG + optional PCA/NMF across sharded .1pz files

#' GPU streaming single-cell pipeline
#'
#' Runs an end-to-end analysis pipeline (log-normalization, HVG selection,
#' optional PCA / NMF) over one or more \code{.1pz} sample directories using
#' the singlet-gpu streaming driver.  The pipeline processes files in chunks of
#' \code{chunk_cols} cells at a time, keeping peak device memory bounded
#' regardless of total cell count.
#'
#' @details
#' \strong{Why not an SCE return value}:
#' The streaming pipeline is designed for inputs that may exceed available RAM
#' (100M+ cells across hundreds of \code{.1pz} shards).  Returning a
#' SingleCellExperiment would require materialising all cells in memory
#' simultaneously, which defeats the purpose of streaming.  The result is
#' therefore a plain \code{list} containing per-gene summaries and optional
#' factor matrices.
#'
#' \strong{PCA limitation}: In-memory PCA is skipped when total cell count
#' exceeds 2M (the \code{int32} NNZ cap in the factornet GPU sparse type).
#' When skipped, \code{$pca_ran} is \code{FALSE} in the result.
#'
#' \strong{Input paths}: each element of \code{input_paths} may be a
#' directory containing \code{counts.1pz} (or other modality files), or a
#' direct path to a \code{.1pz} file.  All inputs must share the same gene
#' axis (rownames are validated across files).
#'
#' @param input_paths   Character vector of \code{.1pz} directory or file paths.
#' @param chunk_cols    Integer.  Cells per streaming chunk; controls peak GPU
#'   memory usage.  Default: \code{100000}.
#' @param run_lognorm  Logical.  Apply per-cell log-normalization.
#'   Default: \code{TRUE}.
#' @param run_hvg      Logical.  Select highly-variable genes.
#'   Default: \code{TRUE}.
#' @param run_pca      Logical.  Run PCA (in-memory; skipped if n_cells > 2M).
#'   Default: \code{FALSE}.
#' @param pca_k        Integer.  Number of PCA components.  Default: \code{50}.
#' @param run_nmf      Logical.  Run chunked NMF.  Default: \code{FALSE}.
#' @param nmf_factors  Integer.  Number of NMF factors.  Default: \code{20}.
#'
#' @return A named \code{list} (NOT a SingleCellExperiment) with:
#' \describe{
#'   \item{\code{$n_cells}}{Double, total cell count across all inputs.}
#'   \item{\code{$n_genes}}{Double, number of genes.}
#'   \item{\code{$n_nnz}}{Double, total non-zero count.}
#'   \item{\code{$size_factors}}{NumericVector \code{[n_cells]} or \code{NULL}.}
#'   \item{\code{$lognorm_target}}{Double, target sum used for normalization.}
#'   \item{\code{$hvg_indices}}{IntegerVector (0-based) or \code{NULL}.}
#'   \item{\code{$hvg_scores}}{NumericVector or \code{NULL}.}
#'   \item{\code{$gene_means}}{NumericVector \code{[n_genes]} or \code{NULL}.}
#'   \item{\code{$gene_vars}}{NumericVector \code{[n_genes]} or \code{NULL}.}
#'   \item{\code{$pca_ran}}{Logical.  \code{TRUE} if PCA was executed.}
#'   \item{\code{$pca_V}}{NumericMatrix \code{[n_cells × pca_k]} or \code{NULL}.}
#'   \item{\code{$pca_U}}{NumericMatrix \code{[n_genes × pca_k]} or \code{NULL}.}
#'   \item{\code{$pca_d}}{NumericVector \code{[pca_k]} or \code{NULL}.}
#'   \item{\code{$nmf_ran}}{Logical.}
#'   \item{\code{$nmf_H}}{NumericMatrix \code{[n_cells × nmf_factors]} or \code{NULL}.}
#'   \item{\code{$nmf_W}}{NumericMatrix \code{[n_genes × nmf_factors]} or \code{NULL}.}
#'   \item{\code{$nmf_d}}{NumericVector \code{[nmf_factors]} or \code{NULL}.}
#'   \item{\code{$nmf_converged}}{Logical or \code{NULL}.}
#'   \item{\code{$nmf_iterations}}{Integer or \code{NULL}.}
#'   \item{\code{$wall_lognorm_s}}{Double, wall-clock seconds for log-norm pass.}
#'   \item{\code{$wall_hvg_s}}{Double.}
#'   \item{\code{$wall_pca_s}}{Double.}
#'   \item{\code{$wall_nmf_s}}{Double.}
#'   \item{\code{$n_chunks_processed}}{Integer.}
#' }
#'
#' @examples
#' \dontrun{
#' paths <- list.files("/mnt/projects/.../quant/scrna/GSE127/",
#'                     recursive = TRUE, pattern = "counts\\.1pz$",
#'                     full.names = TRUE)
#' # Pass parent directories (one per GSM)
#' gsm_dirs <- unique(dirname(paths))
#' result <- run_pipeline(gsm_dirs, chunk_cols = 50000L, run_hvg = TRUE)
#' cat(sprintf("Processed %g cells, %g genes\n", result$n_cells, result$n_genes))
#' }
#'
#' @seealso \code{\link{lognorm}}, \code{\link{hvg}}, \code{\link{run_pca}}
#' @export
run_pipeline <- function(input_paths,
                         chunk_cols   = 100000L,
                         run_lognorm  = TRUE,
                         run_hvg      = TRUE,
                         run_pca      = FALSE,
                         pca_k        = 50L,
                         run_nmf      = FALSE,
                         nmf_factors  = 20L) {

    if (length(input_paths) == 0L) {
        stop("run_pipeline: input_paths must contain at least one path.")
    }
    if (!all(vapply(input_paths, function(p) file.exists(p), logical(1)))) {
        missing <- input_paths[!vapply(input_paths, file.exists, logical(1))]
        stop(sprintf("run_pipeline: path(s) not found: %s",
                     paste(head(missing, 5), collapse = ", ")))
    }

    chunk_cols  <- as.integer(chunk_cols)
    pca_k       <- as.integer(pca_k)
    nmf_factors <- as.integer(nmf_factors)

    if (chunk_cols  < 1L) stop("run_pipeline: chunk_cols must be >= 1.")
    if (pca_k       < 1L) stop("run_pipeline: pca_k must be >= 1.")
    if (nmf_factors < 1L) stop("run_pipeline: nmf_factors must be >= 1.")

    streaming_pipeline_run_cpp(
        input_paths  = as.character(input_paths),
        chunk_cols   = chunk_cols,
        run_lognorm  = isTRUE(run_lognorm),
        run_hvg      = isTRUE(run_hvg),
        run_pca      = isTRUE(run_pca),
        pca_k        = pca_k,
        run_nmf      = isTRUE(run_nmf),
        nmf_factors  = nmf_factors
    )
}
