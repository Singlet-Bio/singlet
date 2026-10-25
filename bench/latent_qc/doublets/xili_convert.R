# Convert the Xi & Li 2021 real-data .rds files (list(count matrix genes x cells, labels "singlet"/"doublet")) to
# <out>/<name>/{matrix.mtx, genes.txt, cells.txt, labels.txt}. Usage: Rscript xili_convert.R IN_DIR OUT_DIR
suppressPackageStartupMessages(library(Matrix))
a <- commandArgs(TRUE); files <- list.files(a[1], pattern = "\.rds$", recursive = TRUE, full.names = TRUE)
for (f in files) {
  nm <- sub("\.rds$", "", basename(f)); o <- file.path(a[2], nm)
  x <- readRDS(f)
  cat("==", nm, class(x), length(x), "\n"); str(x, max.level = 1)
  m <- x[[1]]; lab <- as.character(x[[2]])
  if (!inherits(m, "dgCMatrix")) m <- as(as(m, "CsparseMatrix"), "generalMatrix")
  if (length(lab) != ncol(m)) { cat("label length", length(lab), "!= ncol", ncol(m), "skip\n"); next }
  dir.create(o, showWarnings = FALSE, recursive = TRUE)
  writeMM(m, file.path(o, "matrix.mtx"))
  g <- rownames(m); if (is.null(g)) g <- paste0("g", seq_len(nrow(m)))
  cc <- colnames(m); if (is.null(cc)) cc <- paste0("c", seq_len(ncol(m)))
  writeLines(g, file.path(o, "genes.txt")); writeLines(cc, file.path(o, "cells.txt")); writeLines(lab, file.path(o, "labels.txt"))
  cat(nm, nrow(m), "genes", ncol(m), "cells", sum(lab == "doublet"), "doublets; genes", head(g, 3), "\n")
}
