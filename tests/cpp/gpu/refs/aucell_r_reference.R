#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/aucell_r_reference.R
#
# Reference driver: run AUCell::AUCell_run on a cell × gene expression matrix
# + gene sets, then write per-cell AUC scores to a TSV file.
#
# Environment:
#   R >= 4.0
#   BiocManager::install("AUCell")
#
# Usage:
#   Rscript aucell_r_reference.R \
#       --matrix    <matrix.bin>      # CSC binary (dump_csc.h format, genes×cells)
#       --genesets  <genesets.tsv>    # three-col: set_name<TAB>gene_idx<TAB>weight
#       --output    <auc_scores.tsv>  # output: cells × pathways AUC scores
#       --top-k     500               # aucMaxRank (top-K per cell)
#
# Output TSV:
#   Header row: cell_idx<TAB>set_name_1<TAB>...<TAB>set_name_S
#   Data rows:  int<TAB>float<TAB>...<TAB>float
#   Shape: (n_cells) × (n_sets+1)
#
# The C++ harness writes the input files and reads the output TSV.

suppressPackageStartupMessages({
    if (!requireNamespace("AUCell", quietly = TRUE)) {
        stop("[aucell_r_reference] AUCell not installed. ",
             "Run: BiocManager::install('AUCell')")
    }
    library(AUCell)
})

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
parse_arg <- function(flag, default = NULL) {
    idx <- which(args == flag)
    if (length(idx) == 0) {
        if (is.null(default)) stop("Missing required argument: ", flag)
        return(default)
    }
    return(args[idx + 1L])
}

matrix_path   <- parse_arg("--matrix")
genesets_path <- parse_arg("--genesets")
output_path   <- parse_arg("--output")
top_k         <- as.integer(parse_arg("--top-k", "500"))

# ---------------------------------------------------------------------------
# Load CSC binary (dump_csc.h format)
# Magic = 0x43535343, then n_rows=genes, n_cols=cells (uint32), nnz (uint64),
# float32 values[nnz], int32 indptr[n_cols+1], int32 indices[nnz].
# ---------------------------------------------------------------------------
load_csc_bin <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))

    magic  <- readBin(con, what = "integer", n = 1L, size = 4L, signed = FALSE,
                      endian = "little")
    if (magic != 0x43535343L) stop("Bad magic in CSC binary: ", path)

    n_rows <- readBin(con, what = "integer", n = 1L, size = 4L, signed = FALSE,
                      endian = "little")
    n_cols <- readBin(con, what = "integer", n = 1L, size = 4L, signed = FALSE,
                      endian = "little")
    nnz    <- readBin(con, what = "integer", n = 2L, size = 4L, signed = FALSE,
                      endian = "little")
    # nnz is uint64 stored as two uint32 (lo, hi); assume nnz < 2^32 for simplicity.
    nnz_val <- nnz[1L] + nnz[2L] * 2^32

    values  <- readBin(con, what = "numeric", n = nnz_val, size = 4L,
                       signed = TRUE, endian = "little")
    indptr  <- readBin(con, what = "integer", n = n_cols + 1L, size = 4L,
                       signed = TRUE, endian = "little")
    indices <- readBin(con, what = "integer", n = nnz_val, size = 4L,
                       signed = TRUE, endian = "little")

    # Build sparse Matrix (genes × cells).
    # R uses 1-based indices; CSC binary is 0-based.
    if (!requireNamespace("Matrix", quietly = TRUE))
        stop("Matrix package required")
    mat <- Matrix::sparseMatrix(
        i = indices + 1L,
        p = indptr,
        x = values,
        dims = c(n_rows, n_cols),
        repr = "C"
    )
    rownames(mat) <- as.character(seq_len(n_rows) - 1L)  # gene indices as strings
    colnames(mat) <- as.character(seq_len(n_cols) - 1L)  # cell indices as strings
    message(sprintf("[aucell_r_reference] Loaded CSC: %d genes x %d cells, nnz=%d",
                    n_rows, n_cols, nnz_val))
    mat
}

# ---------------------------------------------------------------------------
# Load gene sets
# ---------------------------------------------------------------------------
gs_df <- read.table(genesets_path, header = FALSE, sep = "\t",
                    col.names = c("set_name", "gene_idx", "weight"),
                    colClasses = c("character", "integer", "numeric"))
if (nrow(gs_df) == 0) stop("[aucell_r_reference] Empty gene sets: ", genesets_path)

gene_sets <- lapply(split(as.character(gs_df$gene_idx), gs_df$set_name),
                    function(x) x)
message(sprintf("[aucell_r_reference] Loaded %d gene sets.", length(gene_sets)))

# ---------------------------------------------------------------------------
# Load matrix
# ---------------------------------------------------------------------------
mat <- load_csc_bin(matrix_path)

# ---------------------------------------------------------------------------
# Run AUCell
# ---------------------------------------------------------------------------
cell_rankings <- AUCell::AUCell_buildRankings(mat, nCores = 1L,
                                               plotStats = FALSE,
                                               verbose = FALSE)
auc_scores    <- AUCell::AUCell_calcAUC(gene_sets, cell_rankings,
                                         aucMaxRank = top_k,
                                         nCores = 1L,
                                         verbose = FALSE)

# auc_scores is an AUCellResults object; extract the AUC matrix (sets × cells).
auc_mat <- AUCell::getAUC(auc_scores)  # n_sets × n_cells matrix

message(sprintf("[aucell_r_reference] AUC matrix: %d sets × %d cells",
                nrow(auc_mat), ncol(auc_mat)))

# ---------------------------------------------------------------------------
# Write output TSV: cell_idx | set1_auc | set2_auc | ...
# ---------------------------------------------------------------------------
n_cells    <- ncol(auc_mat)
set_names  <- rownames(auc_mat)
cell_ids   <- seq_len(n_cells) - 1L   # 0-based cell indices

out_header <- paste(c("cell_idx", set_names), collapse = "\t")
cat(out_header, "\n", sep = "", file = output_path)

for (ci in seq_len(n_cells)) {
    row_vals <- auc_mat[, ci]
    line <- paste(c(cell_ids[ci], sprintf("%.8g", row_vals)), collapse = "\t")
    cat(line, "\n", sep = "", file = output_path, append = TRUE)
}

message(sprintf("[aucell_r_reference] Wrote AUC scores to: %s", output_path))
