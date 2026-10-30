# SoupX, DecontX (celda) and scDblFinder on the called cells of hgmm_1k_v3 (common inputs from prep.py).
suppressMessages({library(Matrix); library(SoupX); library(celda); library(scDblFinder); library(SingleCellExperiment)})
A <- "/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/ambient"; W <- file.path(A, "work")
out <- file.path(A, "results"); dir.create(out, showWarnings = FALSE)
meta <- read.csv(file.path(W, "cells.csv"), stringsAsFactors = FALSE)
genes <- read.csv(file.path(W, "genes.csv"), stringsAsFactors = FALSE)
toc <- readMM(file.path(W, "cells.mtx")); toc <- as(toc, "CsparseMatrix"); rownames(toc) <- genes$id; colnames(toc) <- meta$barcode
rd <- file.path(W, "raw")
tod <- readMM(gzfile(file.path(rd, "matrix.mtx.gz"))); tod <- as(tod, "CsparseMatrix"); rownames(tod) <- genes$id
colnames(tod) <- sub("-1$", "", read.delim(gzfile(file.path(rd, "barcodes.tsv.gz")), header = FALSE)[[1]])
tod <- tod[, colSums(tod) >= 1]   # droplet table SoupX needs; same raw matrix Singlet and CellBender see
log <- list()
save_corrected <- function(m, name) { m <- as(m, "CsparseMatrix"); writeMM(m, file.path(out, paste0(name, "_corrected.mtx"))) }

## ---- SoupX: two cluster definitions (leiden on the cells; oracle = species label) ----
for (cl in c("leiden", "species")) {
  t0 <- Sys.time()
  sc <- SoupChannel(tod, toc)
  cls <- if (cl == "leiden") paste0("L", meta$leiden) else meta$species
  names(cls) <- meta$barcode
  sc <- setClusters(sc, cls[colnames(toc)])
  r <- tryCatch({ sc <- autoEstCont(sc, doPlot = FALSE, contaminationRange = c(0.001, 0.8)); sc$fit$rhoEst }, error = function(e) { message("autoEstCont failed: ", conditionMessage(e)); NA })
  if (!is.na(r)) {
    adj <- adjustCounts(sc, roundToInt = TRUE)
    save_corrected(adj, paste0("soupx_", cl))
  }
  log[[paste0("soupx_", cl)]] <- list(rho = r, seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")))
  cat("SoupX", cl, "rho =", r, "\n")
}

## ---- DecontX: own clustering, and with the empty-droplet background ----
bg <- readMM(file.path(W, "empty.mtx")); bg <- as(bg, "CsparseMatrix"); rownames(bg) <- genes$id
for (v in c("default", "background")) {
  t0 <- Sys.time(); set.seed(1)
  res <- if (v == "default") decontX(toc, verbose = FALSE) else decontX(toc, background = bg, verbose = FALSE)
  contam <- data.frame(barcode = meta$barcode, rho = res$contamination)
  write.csv(contam, file.path(out, paste0("decontx_", v, "_rho.csv")), row.names = FALSE)
  save_corrected(res$decontXcounts, paste0("decontx_", v))
  log[[paste0("decontx_", v)]] <- list(seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), mean_rho = mean(res$contamination))
  cat("DecontX", v, "mean rho =", mean(res$contamination), "\n")
}

## ---- scDblFinder (cluster-based and random modes) ----
sce <- SingleCellExperiment(assays = list(counts = toc))
for (mode in c("default")) {
  set.seed(1); t0 <- Sys.time()
  s <- scDblFinder(sce, verbose = FALSE)
  write.csv(data.frame(barcode = meta$barcode, score = s$scDblFinder.score, call = s$scDblFinder.class), file.path(out, "scdblfinder.csv"), row.names = FALSE)
  log[["scdblfinder"]] <- list(seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")), n_doublets = sum(s$scDblFinder.class == "doublet"))
}
jsonlite_ok <- requireNamespace("jsonlite", quietly = TRUE)
if (jsonlite_ok) writeLines(jsonlite::toJSON(log, auto_unbox = TRUE, pretty = TRUE), file.path(out, "r_tools.json")) else saveRDS(log, file.path(out, "r_tools.rds"))
print(log)
