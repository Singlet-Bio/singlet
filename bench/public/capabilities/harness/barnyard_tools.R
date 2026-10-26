# SPDX-License-Identifier: MIT
# One R competitor per invocation (so the job's /usr/bin/time gives that tool's own wall time and peak memory), at the
# authors' documented defaults, on the inputs written by barnyard_prep.py.
#
#   Rscript barnyard_tools.R soupx|decontx|decontx_bg|scdblfinder WORK_DIR
#
# Output WORK_DIR/tools/<tool>.csv, one row per called cell:
#   barcode, human_after, mouse_after   UMIs per species after the tool's correction (ambient tools)
#   rho                                 the tool's own contamination estimate where it reports one per cell
#   score, call                         doublet score and call (scDblFinder)
suppressMessages({library(Matrix)})
args <- commandArgs(trailingOnly = TRUE)
tool <- args[1]; W <- args[2]
dir.create(file.path(W, "tools"), showWarnings = FALSE)
meta <- read.csv(file.path(W, "cells.csv"), stringsAsFactors = FALSE)
genes <- read.csv(file.path(W, "genes.csv"), stringsAsFactors = FALSE)
toc <- as(readMM(file.path(W, "cells.mtx")), "CsparseMatrix")
rownames(toc) <- genes$id; colnames(toc) <- meta$barcode
mouse <- genes$species == "mouse"
species_after <- function(m) {
  m <- as(m, "CsparseMatrix")
  data.frame(barcode = meta$barcode, human_after = colSums(m[!mouse, , drop = FALSE]), mouse_after = colSums(m[mouse, , drop = FALSE]))
}
info <- list(tool = tool)

if (tool == "soupx") {
  suppressMessages({library(SoupX); library(Seurat)})
  rd <- file.path(W, "raw")
  tod <- as(readMM(gzfile(file.path(rd, "matrix.mtx.gz"))), "CsparseMatrix")
  rownames(tod) <- genes$id
  colnames(tod) <- sub("-1$", "", read.delim(gzfile(file.path(rd, "barcodes.tsv.gz")), header = FALSE)[[1]])
  # SoupX's documented workflow: clusters from a standard Seurat analysis of the cells (defaults), then autoEstCont
  set.seed(1)
  so <- CreateSeuratObject(toc)
  so <- NormalizeData(so, verbose = FALSE); so <- FindVariableFeatures(so, verbose = FALSE); so <- ScaleData(so, verbose = FALSE)
  so <- RunPCA(so, verbose = FALSE); so <- FindNeighbors(so, verbose = FALSE); so <- FindClusters(so, verbose = FALSE)
  cl <- as.character(Idents(so)); names(cl) <- colnames(toc)
  info$n_clusters <- length(unique(cl))
  sc <- SoupChannel(tod, toc)
  sc <- setClusters(sc, cl)
  sc <- tryCatch(autoEstCont(sc, doPlot = FALSE), error = function(e) { message("autoEstCont failed: ", conditionMessage(e)); NULL })
  if (is.null(sc)) { info$failed <- TRUE } else {
    info$rho <- sc$fit$rhoEst
    adj <- adjustCounts(sc)
    out <- species_after(adj[, colnames(toc)]); out$rho <- info$rho
    write.csv(out, file.path(W, "tools", "soupx.csv"), row.names = FALSE)
  }
} else if (tool == "decontx" || tool == "decontx_bg") {
  suppressMessages(library(celda))
  set.seed(1)
  if (tool == "decontx") res <- decontX(toc, verbose = FALSE) else {
    bg <- as(readMM(file.path(W, "empty.mtx")), "CsparseMatrix"); rownames(bg) <- genes$id
    res <- decontX(toc, background = bg, verbose = FALSE)
  }
  out <- species_after(res$decontXcounts); out$rho <- res$contamination
  write.csv(out, file.path(W, "tools", paste0(tool, ".csv")), row.names = FALSE)
  info$mean_rho <- mean(res$contamination)
} else if (tool == "scdblfinder") {
  suppressMessages({library(scDblFinder); library(SingleCellExperiment)})
  set.seed(1)
  s <- scDblFinder(SingleCellExperiment(assays = list(counts = toc)), verbose = FALSE)
  write.csv(data.frame(barcode = meta$barcode, score = s$scDblFinder.score, call = s$scDblFinder.class == "doublet"),
            file.path(W, "tools", "scdblfinder.csv"), row.names = FALSE)
  info$n_called <- sum(s$scDblFinder.class == "doublet")
} else stop("unknown tool ", tool)
writeLines(paste(names(info), unlist(lapply(info, as.character)), sep = "=", collapse = "\n"), file.path(W, "tools", paste0(tool, ".info")))
print(info)
