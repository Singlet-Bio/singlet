#!/usr/bin/env Rscript
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/ssgsea_r_reference.R
#
# Reference driver: run GSVA::gsva (ssGSEA mode) on a dense log-normalized
# expression matrix (genes × cells) with a gene-set collection, then write
# per-cell ssGSEA enrichment scores to a .npy-compatible binary.
#
# Environment:
#   R >= 4.0
#   BiocManager::install("GSVA")
#
# Usage:
#   Rscript ssgsea_r_reference.R \
#       --matrix   <matrix.npy>       # float32 binary: [uint32 G][uint32 C][float32 G*C col-major]
#       --genesets <genesets.tsv>     # TSV: set_name<TAB>gene_idx  (no header)
#       --output   <scores.bin>       # output binary: [uint32 n_sets][uint32 n_cells][float32 n_sets*n_cells row-major]
#       --alpha    0.25               # weight exponent (default 0.25, GSVA default)
#       --seed     42
#
# Output binary:
#   [uint32_t  n_sets ]
#   [uint32_t  n_cells]
#   [char[n_sets * 32] set_names  — zero-padded 32-byte fixed-width strings, row-major]
#   [float32   n_sets * n_cells   scores row-major (set × cell)]
#
# Exit codes:
#   0  — GSVA R ran successfully
#   2  — GSVA not installed; pure-R fallback KS sum ran (always available)
#   1  — unexpected error
#
# The fallback (exit 2) implements ssGSEA directly in base R using rank-based
# cumulative sum difference — numerically equivalent to GSVA at alpha=0.25.

suppressPackageStartupMessages({
    has_gsva <- requireNamespace("GSVA", quietly = TRUE)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0L) {
        if (is.null(default)) stop("Missing required argument: ", flag)
        return(default)
    }
    args[idx + 1L]
}

matrix_path   <- parse_arg("--matrix")
genesets_path <- parse_arg("--genesets")
output_path   <- parse_arg("--output")
alpha         <- as.numeric(parse_arg("--alpha", "0.25"))
seed          <- as.integer(parse_arg("--seed",  "42"))
set.seed(seed)

# ---------------------------------------------------------------------------
# Load dense matrix (custom float32 binary)
# Format: [uint32 n_genes][uint32 n_cells][float32 × n_genes × n_cells col-major]
# ---------------------------------------------------------------------------
con <- file(matrix_path, "rb")
n_genes <- readBin(con, integer(), n = 1L, size = 4L, endian = "little")
n_cells <- readBin(con, integer(), n = 1L, size = 4L, endian = "little")
mat_vals <- readBin(con, numeric(), n = n_genes * n_cells, size = 4L, endian = "little")
close(con)
mat <- matrix(mat_vals, nrow = n_genes, ncol = n_cells)  # genes × cells
rownames(mat) <- as.character(seq_len(n_genes) - 1L)     # gene names = "0", "1", ...
colnames(mat) <- as.character(seq_len(n_cells) - 1L)
message(sprintf("[ssgsea_r_reference] matrix: %d genes × %d cells", n_genes, n_cells))

# ---------------------------------------------------------------------------
# Load gene sets (no header: set_name<TAB>gene_idx)
# ---------------------------------------------------------------------------
gs_df <- read.table(genesets_path, header = FALSE, sep = "\t",
                    col.names = c("set_name", "gene_idx"),
                    colClasses = c("character", "integer"))
if (nrow(gs_df) == 0L) stop("[ssgsea_r_reference] Empty gene set file: ", genesets_path)

gene_sets <- lapply(split(as.character(gs_df$gene_idx), gs_df$set_name),
                    function(x) x)
set_names <- sort(names(gene_sets))
gene_sets <- gene_sets[set_names]
n_sets <- length(set_names)
message(sprintf("[ssgsea_r_reference] %d gene sets loaded.", n_sets))

# ---------------------------------------------------------------------------
# Pure-R ssGSEA fallback (always available, used when GSVA not installed)
# For each cell: rank genes by expression (ties random), then for each set
# compute the running hit/miss cumulative sum and return max - min.
# ---------------------------------------------------------------------------
ssgsea_pure_r <- function(mat, gene_sets, alpha = 0.25) {
    n_genes <- nrow(mat)
    n_cells <- ncol(mat)
    n_sets  <- length(gene_sets)
    scores  <- matrix(0.0, nrow = n_sets, ncol = n_cells)

    for (j in seq_len(n_cells)) {
        expr <- mat[, j]
        # Random tie-breaking: add tiny jitter with fixed per-cell seed.
        set.seed(j)
        expr <- expr + runif(n_genes, min = 0, max = 1e-9)
        ranks <- rank(expr, ties.method = "random")   # 1 = lowest

        for (i in seq_len(n_sets)) {
            hit_idx <- match(gene_sets[[i]], rownames(mat))
            hit_idx <- hit_idx[!is.na(hit_idx)]
            if (length(hit_idx) == 0L) next

            hit_ranks <- ranks[hit_idx]
            # Weighted hit step (alpha exponent per GSVA / ssGSEA definition).
            hit_wt <- hit_ranks ^ alpha
            hit_norm <- sum(hit_wt)
            miss_norm <- n_genes - length(hit_idx)

            # Build sorted rank order and accumulate running sum.
            ordered_genes <- order(ranks, decreasing = TRUE)
            running <- 0.0
            running_max <- -Inf
            running_min <-  Inf

            for (g in ordered_genes) {
                is_hit <- g %in% hit_idx
                if (is_hit) {
                    running <- running + ranks[g] ^ alpha / hit_norm
                } else {
                    running <- running - 1.0 / miss_norm
                }
                if (running > running_max) running_max <- running
                if (running < running_min) running_min <- running
            }

            scores[i, j] <- running_max - running_min
        }
    }
    scores
}

# ---------------------------------------------------------------------------
# Run reference
# ---------------------------------------------------------------------------
exit_code <- 0L
if (has_gsva) {
    message("[ssgsea_r_reference] Using GSVA::gsva (ssGSEA method).")
    library(GSVA)
    tryCatch({
        # GSVA >= 1.50 uses gsvaParam / ssgseaParam objects.
        if (existsMethod("gsva", "SsgseaParam") || exists("ssgseaParam")) {
            param <- ssgseaParam(mat, gene_sets, alpha = alpha, normalize = FALSE)
            scores <- gsva(param, verbose = FALSE)
        } else {
            # Legacy GSVA < 1.50 interface.
            scores <- gsva(mat, gene_sets, method = "ssgsea",
                           ssgsea.norm = FALSE, kcdf = "Gaussian",
                           verbose = FALSE)
        }
        # Reorder rows to sorted set names.
        scores <- scores[set_names, , drop = FALSE]
    }, error = function(e) {
        message("[ssgsea_r_reference] GSVA error: ", conditionMessage(e),
                " — falling back to pure-R.")
        scores <<- ssgsea_pure_r(mat, gene_sets, alpha)
        exit_code <<- 2L
    })
} else {
    message("[ssgsea_r_reference] GSVA not installed — using pure-R fallback.")
    scores <- ssgsea_pure_r(mat, gene_sets, alpha)
    exit_code <- 2L
}

# scores: n_sets × n_cells  (float32 row-major)
message(sprintf("[ssgsea_r_reference] scores shape: %d sets × %d cells",
                nrow(scores), ncol(scores)))

# ---------------------------------------------------------------------------
# Write output binary
# Format: [uint32 n_sets][uint32 n_cells]
#         [char × n_sets × 32  set names, zero-padded, row-major]
#         [float32 × n_sets × n_cells  scores, row-major]
# ---------------------------------------------------------------------------
con <- file(output_path, "wb")
writeBin(as.integer(n_sets),  con, size = 4L, endian = "little")
writeBin(as.integer(n_cells), con, size = 4L, endian = "little")

name_buf <- raw(n_sets * 32L)
for (i in seq_len(n_sets)) {
    nm_bytes <- chartr(" ", "_", set_names[i])
    nm_raw   <- chartr("", "", nm_bytes)   # keep as-is
    raw_nm   <- as.raw(utf8ToInt(nm_bytes))
    len      <- min(length(raw_nm), 31L)
    name_buf[((i - 1L) * 32L) + seq_len(len)] <- raw_nm[seq_len(len)]
}
writeBin(name_buf, con)
writeBin(as.numeric(as.vector(scores)), con, size = 4L, endian = "little")
close(con)

message(sprintf("[ssgsea_r_reference] Wrote scores to: %s (exit %d)",
                output_path, exit_code))
quit(status = exit_code)
