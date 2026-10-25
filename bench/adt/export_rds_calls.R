# SPDX-License-Identifier: MIT
# Export the published per-cell calls of every method from the Howitt et al. 2023 Seurat objects (Zenodo 8304003) to CSV,
# so the benchmark can re-score them with its own F-score code and check it reproduces the paper's Table 1.
#   Rscript bench/adt/export_rds_calls.R <zenodo dir> <out dir>
args <- commandArgs(trailingOnly = TRUE)
src <- args[1]; out <- args[2]
dir.create(out, showWarnings = FALSE, recursive = TRUE)
suppressMessages(library(Seurat))
files <- c(file.path(src, "BAL_data", sprintf("batch%d_all_methods.SEU.rds", 1:3)),
           file.path(src, "cell_line_data", "lmo_all_methods.SEU.rds"))
for (f in files) {
    if (!file.exists(f)) next
    obj <- readRDS(f)
    md <- obj@meta.data
    md$cell <- rownames(md)
    keep <- c("cell", grep("calls$|genetic_donor|Batch|capture|Capture", colnames(md), value = TRUE))
    name <- sub("_all_methods.SEU.rds$", "", basename(f))
    write.csv(md[, unique(keep), drop = FALSE], file.path(out, paste0(name, "_published_calls.csv")), row.names = FALSE)
    cat(name, nrow(md), "cells; columns:", paste(unique(keep), collapse = ", "), "\n")
}
