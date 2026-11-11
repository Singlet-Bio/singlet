# scDblFinder (GPL-3) run as an installed package with default arguments, counts only. Usage: Rscript scdblfinder.R COUNTS.mtx OUT.tsv
suppressMessages({library(Matrix); library(SingleCellExperiment); library(scDblFinder)})
a <- commandArgs(TRUE)
m <- as(readMM(a[1]), "CsparseMatrix")  # genes x cells
set.seed(1)
t0 <- Sys.time()
sce <- scDblFinder(SingleCellExperiment(list(counts = m)), verbose = FALSE)
write.table(data.frame(score = sce$scDblFinder.score), a[2], sep = "\t", row.names = FALSE, quote = FALSE)
cat("scdblfinder seconds", as.numeric(difftime(Sys.time(), t0, units = "secs")), "\n")
