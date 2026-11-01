# S15: the open HTO callers on the HTO count matrix written by pilot_hto.py (cells x tags csv). Rscript pilot_hto_callers.R <hto_counts.csv.gz> <outdir> <libdir>
a <- commandArgs(TRUE); f <- a[1]; out <- a[2]; lib <- a[3]; dir.create(out, showWarnings = FALSE, recursive = TRUE); dir.create(lib, showWarnings = FALSE, recursive = TRUE); .libPaths(c(lib, .libPaths()))
d <- read.csv(gzfile(f), row.names = 1, check.names = FALSE); m <- t(as.matrix(d)); storage.mode(m) <- "numeric"   # tags x cells
m <- m[, colSums(m) > 0, drop = FALSE]; cat("tags", nrow(m), "cells", ncol(m), "\n"); status <- list()
try1 <- function(name, expr) { r <- tryCatch({ force(expr); "ok" }, error = function(e) paste("FAILED:", conditionMessage(e))); status[[name]] <<- r; cat(name, r, "\n") }
try1("hashedDrops", {
  suppressMessages(library(DropletUtils)); h <- hashedDrops(m)
  write.csv(data.frame(barcode = colnames(m), best = rownames(m)[h$Best], second = rownames(m)[h$Second], logfc = h$LogFC, doublet = h$Doublet, confident = h$Confident), file.path(out, "hashedDrops.csv"), row.names = FALSE) })
try1("HTODemux", {
  suppressMessages(library(Seurat)); o <- CreateSeuratObject(counts = m, assay = "HTO"); o <- NormalizeData(o, assay = "HTO", normalization.method = "CLR")
  o <- HTODemux(o, assay = "HTO", positive.quantile = 0.99)
  write.csv(data.frame(barcode = colnames(o), maxid = o$HTO_maxID, global = o$HTO_classification.global, classification = o$HTO_classification), file.path(out, "HTODemux.csv"), row.names = FALSE) })
try1("BFF", {
  if (!requireNamespace("cellhashR", quietly = TRUE)) { if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes", lib = lib, repos = "https://cloud.r-project.org"); remotes::install_github("BimberLab/cellhashR", lib = lib, upgrade = "never", dependencies = TRUE) }
  suppressMessages(library(cellhashR)); cd <- cellhashR::CallAndGenerateReport(m, reportFile = file.path(out, "cellhashR_report.html"), callFile = file.path(out, "cellhashR_calls.csv"), methods = c("bff_raw", "bff_cluster"), doTSNE = FALSE) })
writeLines(paste(names(status), unlist(status), sep = "\t"), file.path(out, "status.tsv"))
