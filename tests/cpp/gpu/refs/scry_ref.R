#!/usr/bin/env Rscript
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/scry_ref.R
#
# Reference implementation: scry::devianceFeatureSelection
# (Townes 2019 — binomial deviance HVG selection)
#
# Activation probe (run before submitting tests that use this script):
#   which Rscript && Rscript -e 'library(scry); cat("scry OK\n")' 2>&1
#
# Usage:
#   Rscript scry_ref.R <input_mtx> <output_csv> [use_poisson]
#
# Arguments:
#   input_mtx   - path to a Matrix Market (.mtx) file; rows=genes, cols=cells;
#                 integer raw counts (no log-transform).
#   output_csv  - path to write per-gene deviance values (CSV, header "deviance").
#   use_poisson - optional "TRUE" or "FALSE" (default FALSE = binomial).
#
# Output CSV format:
#   gene_idx,deviance
#   0,1234.56
#   1,789.01
#   ...
#   (0-indexed gene index, matching the row order in input_mtx)
#
# Exit code: 0 on success, 1 on error.
#
# Required R packages: Matrix, SingleCellExperiment, scry
# Install on g001/g008:
#   Rscript -e 'if (!requireNamespace("BiocManager",quietly=TRUE)) install.packages("BiocManager"); BiocManager::install("scry")'

suppressPackageStartupMessages({
  library(Matrix)
  library(SingleCellExperiment)
  library(scry)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript scry_ref.R <input_mtx> <output_csv> [use_poisson]\n",
      file = stderr())
  quit(status = 1)
}

input_mtx   <- args[1]
output_csv  <- args[2]
use_poisson <- if (length(args) >= 3) (toupper(args[3]) == "TRUE") else FALSE

cat(sprintf("[scry_ref.R] loading: %s\n", input_mtx))
cat(sprintf("[scry_ref.R] use_poisson: %s\n", use_poisson))

# Read Matrix Market (rows = genes, cols = cells)
mat <- readMM(input_mtx)
cat(sprintf("[scry_ref.R] matrix: %d genes x %d cells, nnz=%d\n",
    nrow(mat), ncol(mat), nnzero(mat)))

n_genes <- nrow(mat)
n_cells <- ncol(mat)

# Build SingleCellExperiment with counts assay
sce <- SingleCellExperiment(assays = list(counts = mat))

# Run scry::devianceFeatureSelection
# Returns SCE with rowData(sce)$deviance added.
# use_assay="counts" is the default; fam= controls binomial vs Poisson.
fam <- if (use_poisson) "poisson" else "binomial"
cat(sprintf("[scry_ref.R] running devianceFeatureSelection(fam='%s')...\n", fam))

sce <- devianceFeatureSelection(sce, assay = "counts", fam = fam)

dev_vec <- rowData(sce)$deviance
cat(sprintf("[scry_ref.R] deviance: length=%d, min=%.6f, max=%.6f, mean=%.6f\n",
    length(dev_vec), min(dev_vec, na.rm=TRUE),
    max(dev_vec, na.rm=TRUE), mean(dev_vec, na.rm=TRUE)))

# Write output: 0-indexed gene_idx, deviance value
df <- data.frame(
  gene_idx = seq(0L, n_genes - 1L),
  deviance = as.numeric(dev_vec)
)
write.csv(df, output_csv, row.names = FALSE, quote = FALSE)
cat(sprintf("[scry_ref.R] wrote %d deviance values to: %s\n", n_genes, output_csv))
