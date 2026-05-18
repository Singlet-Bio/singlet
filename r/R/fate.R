# SPDX-License-Identifier: MIT
# singlet-gpu/r/R/fate.R
#
# GPU cell fate inference:
#   run_cospar()    — transition map from lineage barcodes (cycle 41)
#   run_cellrank2() — absorption probabilities from transition matrix (cycle 43)
#   run_palantir()  — diffusion pseudotime + branch probabilities (cycle 45)
#
# All three functions take a SingleCellExperiment and return an enriched SCE.
# Closest published CPU analogs: Cospar (Python), CellRank 2 (Python), Palantir (Python).


# ---------------------------------------------------------------------------
# run_cospar
# ---------------------------------------------------------------------------

#' GPU cell fate transition mapping (Cospar)
#'
#' Infers finite-time fate transition probabilities from clonal lineage barcodes
#' using GPU-accelerated constrained coordinate descent.  Mirrors the \href{
#' https://github.com/KlugerLab/cospar}{Cospar} Python package (Nature Biotech 2022)
#' but runs the entire optimization on GPU.
#'
#' @details
#' The algorithm:
#' \enumerate{
#'   \item Builds a state-similarity kernel from PCA kNN graph.
#'   \item Builds a lineage coupling matrix from shared clone IDs across time points.
#'   \item Optimises the transition map T via 50-iteration coordinate descent with
#'     L1 + state-similarity regularisation.
#'   \item Derives per-cell fate bias, potency score, and driver genes.
#' }
#'
#' Results are stored as:
#' \itemize{
#'   \item \code{colData(sce)$cospar_potency} — scalar lineage entropy per cell.
#'   \item \code{reducedDim(sce, "cospar_fate_bias")} — cells × n_fates matrix.
#'   \item \code{rowData(sce)$cospar_driver_{1..n_fates}} — driver gene scores.
#'   \item \code{metadata(sce)$cospar} — full result list.
#' }
#'
#' @param sce        A \code{\link[SingleCellExperiment]{SingleCellExperiment}}.
#' @param assay_name Character.  Assay to use.  Default: \code{"logcounts"}.
#' @param time_col   Character.  \code{colData} column with time-point labels
#'   (0-based integer or factor).  Default: \code{"time_point"}.
#' @param clone_col  Character.  \code{colData} column with clone IDs (integer;
#'   \code{-1} = unassigned).  Set to \code{NULL} to use state-only mode.
#'   Default: \code{"clone_id"}.
#' @param n_fates    Integer.  Number of terminal fates.  \code{0L} = auto-detect.
#'   Default: \code{0L}.
#' @param lambda1    Numeric.  L1 sparsity weight.  Default: \code{0.05}.
#' @param lambda2    Numeric.  State-similarity weight.  Default: \code{0.2}.
#' @param max_iter   Integer.  Coordinate-descent iterations.  Default: \code{50L}.
#' @param tol        Numeric.  Relative Frobenius convergence threshold.
#'   Default: \code{1e-4}.
#' @param seed       Integer.  RNG seed.  Default: \code{0L}.
#' @param resource   Character.  \code{"gpu"} / \code{"cpu"} / \code{"auto"}.
#'   Default: \code{"auto"}.
#'
#' @return The input \code{sce} with fate inference results stored in
#'   \code{colData}, \code{reducedDim}, \code{rowData}, and \code{metadata}.
#'
#' @seealso \code{\link{run_cellrank2}}, \code{\link{run_palantir}}
#' @export
run_cospar <- function(sce,
                       assay_name = "logcounts",
                       time_col   = "time_point",
                       clone_col  = "clone_id",
                       n_fates    = 0L,
                       lambda1    = 0.05,
                       lambda2    = 0.2,
                       max_iter   = 50L,
                       tol        = 1e-4,
                       seed       = 0L,
                       resource   = "auto") {

    .assert_sce(sce, "run_cospar")

    if (!assay_name %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_cospar: assay '%s' not found.", assay_name))
    if (!time_col %in% names(SummarizedExperiment::colData(sce)))
        stop(sprintf("run_cospar: colData column '%s' not found.", time_col))

    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    tp_raw <- SummarizedExperiment::colData(sce)[[time_col]]
    if (is.factor(tp_raw)) tp_raw <- as.integer(tp_raw) - 1L
    time_points <- as.integer(tp_raw)

    if (!is.null(clone_col) && clone_col %in% names(SummarizedExperiment::colData(sce))) {
        clone_ids <- as.integer(SummarizedExperiment::colData(sce)[[clone_col]])
    } else {
        clone_ids <- integer(0L)  # state-only mode
    }

    res <- cospar_cpp(mat, time_points, clone_ids,
                      as.integer(n_fates),
                      as.double(lambda1), as.double(lambda2),
                      as.integer(max_iter), as.double(tol),
                      as.integer(seed), as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_cospar: %s", res$error))

    # Store potency in colData
    cd <- SummarizedExperiment::colData(sce)
    cd$cospar_potency <- res$potency_score
    SummarizedExperiment::colData(sce) <- cd

    # Store fate bias as a reducedDim
    SingleCellExperiment::reducedDim(sce, "cospar_fate_bias") <- res$fate_bias

    # Store metadata
    SummarizedExperiment::metadata(sce)$cospar <- list(
        fate_bias       = res$fate_bias,
        potency_score   = res$potency_score,
        driver_genes    = res$driver_genes,
        iters_run       = res$iters_run,
        converged       = res$converged,
        state_only_mode = res$state_only_mode
    )

    sce
}


# ---------------------------------------------------------------------------
# run_cellrank2
# ---------------------------------------------------------------------------

#' GPU absorption probability computation (CellRank 2)
#'
#' Computes per-cell absorption probabilities for terminal states from a
#' row-stochastic transition matrix via GPU-batched GMRES.  Mirrors
#' \href{https://github.com/theislab/cellrank}{CellRank 2} (Nature Methods 2024).
#'
#' @details
#' The transition matrix may come from:
#' \itemize{
#'   \item \code{metadata(sce)$cospar$transition_map} (cycle 41 Cospar).
#'   \item RNA velocity moments (cycle 13 \code{compute_velocity_moments}).
#'   \item A kNN graph-based Markov chain.
#' }
#'
#' Results are stored as:
#' \itemize{
#'   \item \code{reducedDim(sce, "cellrank2_absorption")} — cells × n_terminals.
#'   \item \code{metadata(sce)$cellrank2} — full result list.
#' }
#'
#' @param sce               A \code{SingleCellExperiment}.
#' @param transition_matrix A \code{dgCMatrix} [n_cells × n_cells], row-stochastic.
#'   If \code{NULL}, attempts to extract from \code{metadata(sce)$cospar$transition_map}.
#' @param terminal_states   IntegerVector (0-based) or character vector of cell names.
#'   If character, matched against \code{colnames(sce)}.
#' @param max_iter          Integer.  GMRES restart dimension.  Default: \code{50L}.
#' @param tol               Numeric.  Relative residual tolerance.  Default: \code{1e-6}.
#' @param seed              Integer.  Default: \code{0L}.
#' @param resource          Character.  Default: \code{"auto"}.
#'
#' @return The input \code{sce} enriched with absorption probabilities.
#'
#' @seealso \code{\link{run_cospar}}, \code{\link{run_palantir}}
#' @export
run_cellrank2 <- function(sce,
                          transition_matrix = NULL,
                          terminal_states,
                          max_iter  = 50L,
                          tol       = 1e-6,
                          seed      = 0L,
                          resource  = "auto") {

    .assert_sce(sce, "run_cellrank2")

    if (is.null(transition_matrix)) {
        cospar_meta <- SummarizedExperiment::metadata(sce)$cospar
        if (!is.null(cospar_meta$transition_map)) {
            transition_matrix <- cospar_meta$transition_map
        } else {
            stop("run_cellrank2: provide transition_matrix or run run_cospar() first.")
        }
    }
    if (!methods::is(transition_matrix, "dgCMatrix"))
        transition_matrix <- methods::as(transition_matrix, "dgCMatrix")

    # Resolve terminal_states to 0-based integer indices
    if (is.character(terminal_states)) {
        cn <- colnames(sce)
        if (is.null(cn))
            stop("run_cellrank2: character terminal_states requires colnames(sce).")
        idx <- match(terminal_states, cn)
        if (any(is.na(idx)))
            stop("run_cellrank2: some terminal_states not found in colnames(sce).")
        term_ids <- as.integer(idx) - 1L
    } else {
        term_ids <- as.integer(terminal_states)
    }

    res <- cellrank2_cpp(transition_matrix, term_ids,
                         as.integer(max_iter), as.double(tol),
                         as.integer(seed), as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_cellrank2: %s", res$error))

    SingleCellExperiment::reducedDim(sce, "cellrank2_absorption") <- res$absorption_prob

    SummarizedExperiment::metadata(sce)$cellrank2 <- list(
        absorption_prob = res$absorption_prob,
        converged_all   = res$converged_all,
        n_terminals     = res$n_terminals
    )

    sce
}


# ---------------------------------------------------------------------------
# run_palantir
# ---------------------------------------------------------------------------

#' GPU diffusion pseudotime + branch probabilities (Palantir)
#'
#' Builds a diffusion operator from scratch, then computes pseudotime and
#' per-branch fate probabilities via GPU-accelerated spectral decomposition +
#' random walk.  Mirrors \href{https://github.com/dpeerlab/Palantir}{Palantir}
#' (Nature Biotech 2019).
#'
#' @details
#' Unlike \code{\link{run_cospar}} (requires lineage barcodes) and
#' \code{\link{run_cellrank2}} (requires a pre-built transition matrix),
#' \code{run_palantir} works on any log-normalized scRNA dataset with a
#' specified starting cell.
#'
#' Results are stored as:
#' \itemize{
#'   \item \code{colData(sce)$palantir_pseudotime} — scalar pseudotime [0, 1].
#'   \item \code{reducedDim(sce, "palantir_branch_probs")} — cells × n_terminals.
#'   \item \code{metadata(sce)$palantir} — full result list.
#' }
#'
#' @param sce                    A \code{SingleCellExperiment}.
#' @param assay_name             Character.  Default: \code{"logcounts"}.
#' @param start_cell             Integer (0-based) or character (cell name).
#'   The root (most stem-like) cell.
#' @param n_terminals            Integer.  Number of terminal states.
#'   \code{0L} = auto-detect.  Default: \code{0L}.
#' @param n_neighbors            Integer.  kNN graph k.  Default: \code{30L}.
#' @param n_diffusion_components Integer.  Spectral dimensions.  Default: \code{10L}.
#' @param seed                   Integer.  Default: \code{0L}.
#' @param resource               Character.  Default: \code{"auto"}.
#'
#' @return The input \code{sce} enriched with pseudotime and branch probabilities.
#'
#' @seealso \code{\link{run_cospar}}, \code{\link{run_cellrank2}}
#' @export
run_palantir <- function(sce,
                         assay_name             = "logcounts",
                         start_cell,
                         n_terminals            = 0L,
                         n_neighbors            = 30L,
                         n_diffusion_components = 10L,
                         seed                   = 0L,
                         resource               = "auto") {

    .assert_sce(sce, "run_palantir")

    if (!assay_name %in% SummarizedExperiment::assayNames(sce))
        stop(sprintf("run_palantir: assay '%s' not found.", assay_name))

    mat <- SummarizedExperiment::assay(sce, assay_name)
    if (!methods::is(mat, "dgCMatrix")) mat <- methods::as(mat, "dgCMatrix")

    # Resolve start_cell
    if (is.character(start_cell)) {
        cn <- colnames(sce)
        if (is.null(cn)) stop("run_palantir: character start_cell requires colnames(sce).")
        idx <- match(start_cell, cn)
        if (is.na(idx)) stop(sprintf("run_palantir: start_cell '%s' not in colnames.", start_cell))
        start_idx <- as.integer(idx) - 1L
    } else {
        start_idx <- as.integer(start_cell)
    }

    res <- palantir_cpp(mat, start_idx,
                        as.integer(n_terminals),
                        as.integer(n_neighbors),
                        as.integer(n_diffusion_components),
                        as.integer(seed), as.character(resource))

    if (!is.null(res$error))
        stop(sprintf("run_palantir: %s", res$error))

    cd <- SummarizedExperiment::colData(sce)
    cd$palantir_pseudotime <- res$pseudotime
    SummarizedExperiment::colData(sce) <- cd

    if (ncol(res$branch_probs) > 0)
        SingleCellExperiment::reducedDim(sce, "palantir_branch_probs") <- res$branch_probs

    SummarizedExperiment::metadata(sce)$palantir <- list(
        pseudotime     = res$pseudotime,
        branch_probs   = res$branch_probs,
        terminal_cells = res$terminal_cells,
        n_terminals    = res$n_terminals
    )

    sce
}
