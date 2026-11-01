# S15: SoupX, DecontX and scDblFinder on STARsolo's GeneFull filtered cells (raw droplets for SoupX). Rscript pilot_r.R <Solo.out/GeneFull> <outdir>
suppressMessages({library(Matrix); library(SoupX); library(celda); library(scDblFinder); library(SingleCellExperiment); library(irlba)})
a <- commandArgs(TRUE); d <- a[1]; out <- a[2]; dir.create(out, showWarnings = FALSE, recursive = TRUE)
rd <- function(p) { f <- function(n) { x <- file.path(p, n); if (file.exists(x)) x else paste0(x, ".gz") }
  m <- as(readMM(f("matrix.mtx")), "CsparseMatrix"); b <- read.delim(f("barcodes.tsv"), header = FALSE)[[1]]
  g <- read.delim(f("features.tsv"), header = FALSE)[[1]]; dimnames(m) <- list(g, b); m }
toc <- rd(file.path(d, "filtered")); tod <- rd(file.path(d, "raw")); tod <- tod[, colSums(tod) >= 1]
cat("cells", ncol(toc), "droplets", ncol(tod), "\n"); log <- list(n_cells = ncol(toc))
# simple clusters for SoupX: 20 PCs of log-normalised top-2000 variable genes, k-means
ln <- log1p(t(t(toc) / pmax(colSums(toc), 1) * 1e4)); v <- Matrix::rowMeans(ln^2) - Matrix::rowMeans(ln)^2
hv <- order(v, decreasing = TRUE)[1:min(2000, nrow(ln))]
pc <- irlba(t(ln[hv, ]), nv = 20)$u; set.seed(1); cl <- kmeans(pc, centers = min(10, ncol(toc) - 1), nstart = 3)$cluster; names(cl) <- colnames(toc)
r <- tryCatch({ sc <- SoupChannel(tod, toc); sc <- setClusters(sc, cl); sc <- autoEstCont(sc, doPlot = FALSE, contaminationRange = c(0.001, 0.8)); sc$fit$rhoEst },
              error = function(e) { message("SoupX failed: ", conditionMessage(e)); NA })
log$soupx_rho <- r; cat("SoupX rho", r, "\n")
res <- tryCatch({ set.seed(1); decontX(toc, verbose = FALSE) }, error = function(e) { message("decontX failed: ", conditionMessage(e)); NULL })
if (!is.null(res)) { write.csv(data.frame(barcode = colnames(toc), rho = res$contamination), file.path(out, "decontx_rho.csv"), row.names = FALSE); log$decontx_mean_rho <- mean(res$contamination) }
s <- tryCatch({ set.seed(1); scDblFinder(SingleCellExperiment(assays = list(counts = toc)), verbose = FALSE) }, error = function(e) { message("scDblFinder failed: ", conditionMessage(e)); NULL })
if (!is.null(s)) { write.csv(data.frame(barcode = colnames(toc), score = s$scDblFinder.score, class = s$scDblFinder.class), file.path(out, "scdblfinder.csv"), row.names = FALSE); log$scdblfinder_doublets <- sum(s$scDblFinder.class == "doublet") }
writeLines(paste(names(log), unlist(log), sep = "\t"), file.path(out, "summary.tsv"))
