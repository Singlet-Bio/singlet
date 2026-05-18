#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/milo_r_reference.R
#
# Reference driver for Milo differential abundance testing.
# Primary reference: miloR (Dann et al. 2022, Nature Biotechnology).
# Fallback (always available): pure-R NB GLM per neighborhood via MASS::glm.nb.
#
# The driver:
#   1. Reads a pre-built neighborhood count matrix (n_nh x n_donors) from TSV.
#   2. Reads donor metadata (donor_id, condition, offset columns).
#   3. Optionally reads a neighborhood adjacency TSV (for graph-weighted FDR).
#   4. Runs miloR::testNhoods() if miloR is installed, else falls back to
#      MASS::glm.nb per neighborhood.
#   5. Writes per-neighborhood results: log_fc, se, p_value, fdr.
#
# Environment:
#   R >= 4.0
#   Bioconductor: BiocManager::install("miloR")
#   CRAN: install.packages(c("MASS", "igraph", "SingleCellExperiment", "scater"))
#
# Usage:
#   Rscript milo_r_reference.R \
#       --nh_counts    <nh_counts.tsv>     # n_nh x n_donors (integer), tab-separated
#       --donor_meta   <donor_meta.tsv>    # columns: donor_id condition offset
#       --out_dir      <dir>               # writes milo_results.tsv
#       [--nh_adj      <nh_adj.tsv>]       # optional: 3-col (nh_i, nh_j, 1) for graph FDR
#       [--seed        <int>]              # default: 42
#
# Output: out_dir/milo_results.tsv
#   columns: nh_idx  log_fc  se  p_value  fdr  converged  backend
#
# Exit codes:
#   0  — success
#   1  — argument or I/O error
#   2  — neither miloR nor MASS available (treated as "ref unavailable" by C++ test)
#   3  — data shape error (< 2 donors per condition, cannot run NB GLM)

suppressPackageStartupMessages({
    has_mass <- requireNamespace("MASS", quietly = TRUE)
    has_milo <- requireNamespace("miloR", quietly = TRUE) &&
                requireNamespace("SingleCellExperiment", quietly = TRUE)
    if (!has_mass && !has_milo) {
        message("[milo_r_reference] Neither miloR nor MASS is installed.")
        message("  Install fallback: install.packages('MASS')")
        message("  Install primary:  BiocManager::install('miloR')")
        quit(save = "no", status = 2L)
    }
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0L) {
        if (is.null(default)) {
            message("[milo_r_reference] Missing required argument: ", flag)
            quit(save = "no", status = 1L)
        }
        return(default)
    }
    return(args[idx + 1L])
}

nh_counts_path <- parse_arg("--nh_counts")
donor_meta_path <- parse_arg("--donor_meta")
out_dir         <- parse_arg("--out_dir")
nh_adj_path     <- parse_arg("--nh_adj", NULL)
seed_val        <- as.integer(parse_arg("--seed", "42"))

set.seed(seed_val)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message(sprintf("[milo_r_reference] nh_counts=%s  donor_meta=%s  out_dir=%s",
                nh_counts_path, donor_meta_path, out_dir))

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
nh_counts_mat <- as.matrix(read.table(nh_counts_path, header = FALSE, sep = "\t"))
donor_meta    <- read.table(donor_meta_path,  header = TRUE,  sep = "\t",
                             stringsAsFactors = FALSE)

n_nh     <- nrow(nh_counts_mat)
n_donors <- ncol(nh_counts_mat)

if (nrow(donor_meta) != n_donors) {
    message(sprintf("[milo_r_reference] donor_meta rows (%d) != n_donors (%d)",
                    nrow(donor_meta), n_donors))
    quit(save = "no", status = 1L)
}

message(sprintf("[milo_r_reference] n_nh=%d  n_donors=%d", n_nh, n_donors))

# Validate minimum donors per condition
cond_table <- table(donor_meta$condition)
if (any(cond_table < 2)) {
    message("[milo_r_reference] Fewer than 2 donors in some condition — cannot fit NB GLM.")
    quit(save = "no", status = 3L)
}

# ---------------------------------------------------------------------------
# Helper: run MASS::glm.nb on one neighborhood row
# ---------------------------------------------------------------------------
fit_one_nh <- function(nh_row, donor_meta) {
    df <- data.frame(
        count     = as.integer(nh_row),
        condition = factor(donor_meta$condition),
        offset    = log(pmax(donor_meta$offset, 1))
    )
    result <- tryCatch({
        fit <- MASS::glm.nb(count ~ condition + offset(offset), data = df,
                            control = glm.control(maxit = 20, epsilon = 1e-6))
        coefs   <- summary(fit)$coefficients
        # Extract the condition coefficient (second row if binary)
        cond_row <- coefs[grepl("^condition", rownames(coefs)), , drop = FALSE]
        if (nrow(cond_row) == 0L) {
            return(list(log_fc = NA_real_, se = NA_real_, p_value = NA_real_,
                        converged = FALSE))
        }
        log_fc <- cond_row[1, "Estimate"]
        se_val <- cond_row[1, "Std. Error"]
        pval   <- cond_row[1, "Pr(>|z|)"]
        list(log_fc = log_fc, se = se_val, p_value = pval, converged = fit$converged)
    }, error = function(e) {
        list(log_fc = NA_real_, se = NA_real_, p_value = NA_real_, converged = FALSE)
    })
    result
}

# ---------------------------------------------------------------------------
# Attempt miloR path first; fall back to MASS::glm.nb
# ---------------------------------------------------------------------------
backend <- "MASS_glm.nb"
results_df <- NULL

if (requireNamespace("miloR", quietly = TRUE) &&
    requireNamespace("SingleCellExperiment", quietly = TRUE)) {
    tryCatch({
        suppressPackageStartupMessages({
            library(miloR)
            library(SingleCellExperiment)
        })
        message("[milo_r_reference] Using miloR backend")

        # miloR requires a Milo object built from a SingleCellExperiment.
        # Since we start from pre-computed nh_counts, we wrap directly into
        # testNhoods() by constructing a minimal Milo-like structure.
        # We use a dummy reducedDim (PCA on nh_counts transpose) to satisfy SCE.
        dummy_expr  <- t(nh_counts_mat)  # n_donors x n_nh
        sce         <- SingleCellExperiment(
            assays = list(counts = dummy_expr)
        )
        # Attach column metadata for donors
        colData(sce)$donor_id  <- donor_meta$donor_id
        colData(sce)$condition <- factor(donor_meta$condition)
        colData(sce)$offset    <- donor_meta$offset

        milo_obj <- Milo(sce)

        # nhoodCounts is n_nh x n_donors
        nhoodCounts(milo_obj) <- nh_counts_mat

        # If adjacency provided, build nhoodGraph
        if (!is.null(nh_adj_path) && file.exists(nh_adj_path)) {
            adj_df <- read.table(nh_adj_path, header = FALSE, sep = "\t")
            adj_mat <- Matrix::sparseMatrix(
                i = adj_df[, 1] + 1L,
                j = adj_df[, 2] + 1L,
                x = rep(1.0, nrow(adj_df)),
                dims = c(n_nh, n_nh),
                symmetric = FALSE
            )
            nhoodGraph(milo_obj) <- igraph::graph_from_adjacency_matrix(
                adj_mat, mode = "undirected", weighted = TRUE
            )
        } else {
            # Build trivial chain graph: NH i connected to i+1
            if (n_nh > 1L) {
                nhoodGraph(milo_obj) <- igraph::make_ring(n_nh, directed = FALSE)
            } else {
                nhoodGraph(milo_obj) <- igraph::make_empty_graph(n = 1L, directed = FALSE)
            }
        }

        # Design matrix
        design_df <- as.data.frame(colData(milo_obj)[, c("condition", "offset")])
        design_df$condition <- factor(design_df$condition)

        milo_res <- testNhoods(
            milo_obj,
            design = ~ condition,
            design.df = design_df,
            reduced.dim = NULL
        )

        results_df <- data.frame(
            nh_idx    = seq_len(n_nh) - 1L,
            log_fc    = milo_res$logFC,
            se        = milo_res$SE,
            p_value   = milo_res$PValue,
            fdr       = milo_res$FDR,
            converged = TRUE,
            backend   = "miloR"
        )
        backend <- "miloR"
        message(sprintf("[milo_r_reference] miloR done: %d neighborhoods", n_nh))
    }, error = function(e) {
        message("[milo_r_reference] miloR failed (", conditionMessage(e),
                "); falling back to MASS::glm.nb")
        results_df <<- NULL
    })
}

# ---------------------------------------------------------------------------
# MASS::glm.nb fallback
# ---------------------------------------------------------------------------
if (is.null(results_df)) {
    message("[milo_r_reference] Using MASS::glm.nb fallback")
    library(MASS)

    log_fc_vec    <- numeric(n_nh)
    se_vec        <- numeric(n_nh)
    pval_vec      <- numeric(n_nh)
    converged_vec <- logical(n_nh)

    for (i in seq_len(n_nh)) {
        res <- fit_one_nh(nh_counts_mat[i, ], donor_meta)
        log_fc_vec[i]    <- if (is.null(res$log_fc))   NA_real_ else res$log_fc
        se_vec[i]        <- if (is.null(res$se))        NA_real_ else res$se
        pval_vec[i]      <- if (is.null(res$p_value))   NA_real_ else res$p_value
        converged_vec[i] <- if (is.null(res$converged)) FALSE    else res$converged
    }

    # BH FDR (non-graph-weighted since we have no graph in fallback)
    valid_p          <- !is.na(pval_vec)
    fdr_vec          <- rep(NA_real_, n_nh)
    fdr_vec[valid_p] <- p.adjust(pval_vec[valid_p], method = "BH")

    results_df <- data.frame(
        nh_idx    = seq_len(n_nh) - 1L,
        log_fc    = log_fc_vec,
        se        = se_vec,
        p_value   = pval_vec,
        fdr       = fdr_vec,
        converged = converged_vec,
        backend   = "MASS_glm.nb"
    )
    message(sprintf("[milo_r_reference] MASS::glm.nb done: %d neighborhoods, "
                    "%d converged",
                    n_nh, sum(converged_vec, na.rm = TRUE)))
}

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
out_path <- file.path(out_dir, "milo_results.tsv")
write.table(results_df, file = out_path, sep = "\t",
            row.names = FALSE, quote = FALSE)
message(sprintf("[milo_r_reference] Wrote %s", out_path))
quit(save = "no", status = 0L)
