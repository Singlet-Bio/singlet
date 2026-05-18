#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/scran_ref.R
#
# Reference implementation: scran::computeSumFactors
#
# Usage:
#   Rscript scran_ref.R <input_mtx> <output_csv> [sizes]
#
# Arguments:
#   input_mtx  - path to a Matrix Market file (.mtx); rows = genes, cols = cells.
#                The file is assumed to be integer counts (no log-transform).
#   output_csv - path to write size factors (one per line, CSV with header "sf").
#   sizes      - optional comma-separated pool sizes, default "21,41,61,81,101"
#
# Output CSV format:
#   sf
#   1.0234
#   0.9871
#   ...
#
# Exit code: 0 on success, 1 on error (also writes an error message to stderr).

suppressPackageStartupMessages({
  library(Matrix)
  library(scran)
  library(SingleCellExperiment)
  library(BiocParallel)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript scran_ref.R <input_mtx> <output_csv> [pool_sizes]\n",
      file = stderr())
  quit(status = 1)
}

input_mtx  <- args[1]
output_csv <- args[2]
sizes_str  <- if (length(args) >= 3) args[3] else "21,41,61,81,101"

# Parse pool sizes
sizes <- as.integer(strsplit(sizes_str, ",")[[1]])
cat(sprintf("[scran_ref.R] loading: %s\n", input_mtx))
cat(sprintf("[scran_ref.R] pool sizes: %s\n", paste(sizes, collapse=",")))

# Read Matrix Market (rows = genes, cols = cells)
mat <- readMM(input_mtx)
cat(sprintf("[scran_ref.R] matrix: %d genes x %d cells, nnz=%d\n",
    nrow(mat), ncol(mat), nnzero(mat)))

n_cells <- ncol(mat)

# Clamp sizes to those with at least one full pool possible given n_cells.
# scran requires min(sizes) <= floor(n_cells / 2).
max_size_allowed <- max(1L, floor(n_cells / 2L))
sizes_use <- sizes[sizes <= max_size_allowed]
if (length(sizes_use) == 0) {
  # Fall back to a single minimal pool
  sizes_use <- max(5L, min(sizes))
  sizes_use <- min(sizes_use, max_size_allowed)
  cat(sprintf("[scran_ref.R] WARNING: all requested sizes exceed n_cells/2=%d; "
              "using sizes=%d\n", max_size_allowed, sizes_use))
}

# Build SingleCellExperiment
sce <- SingleCellExperiment(assays = list(counts = mat))

# Run computeSumFactors with BPPARAM=SerialParam for reproducibility
BPPARAM <- SerialParam()
sce <- computeSumFactors(sce, sizes = sizes_use, BPPARAM = BPPARAM)

sf <- sizeFactors(sce)

# Write output CSV
df <- data.frame(sf = sf)
write.csv(df, output_csv, row.names = FALSE, quote = FALSE)
cat(sprintf("[scran_ref.R] wrote %d size factors to: %s\n", length(sf), output_csv))
cat(sprintf("[scran_ref.R] median SF=%.6f, min=%.6f, max=%.6f\n",
    median(sf), min(sf), max(sf)))
