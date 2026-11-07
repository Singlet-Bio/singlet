# S15: DropletUtils emptyDrops on STARsolo's raw GeneFull matrix. Rscript pilot_edrops.R <Solo.out/GeneFull/raw> <outdir> <libdir>
a <- commandArgs(TRUE); p <- a[1]; out <- a[2]; lib <- a[3]; dir.create(out, showWarnings = FALSE, recursive = TRUE); dir.create(lib, showWarnings = FALSE, recursive = TRUE)
.libPaths(c(lib, .libPaths()))
if (!requireNamespace("DropletUtils", quietly = TRUE)) { if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = lib, repos = "https://cloud.r-project.org"); BiocManager::install("DropletUtils", lib = lib, update = FALSE, ask = FALSE) }
suppressMessages({library(DropletUtils); library(Matrix)})
f <- function(n) { x <- file.path(p, n); if (file.exists(x)) x else paste0(x, ".gz") }
m <- as(readMM(f("matrix.mtx")), "CsparseMatrix"); b <- read.delim(f("barcodes.tsv"), header = FALSE)[[1]]; dimnames(m) <- list(NULL, b)
m <- m[, colSums(m) > 0]; set.seed(100)
e <- emptyDrops(m, lower = 100, niters = 10000)
keep <- !is.na(e$FDR) & e$FDR <= 0.001
write.csv(data.frame(barcode = colnames(m), total = e$Total, logprob = e$LogProb, fdr = e$FDR)[keep, ], file.path(out, "emptydrops_cells.csv"), row.names = FALSE)
writeLines(c(paste("droplets", ncol(m)), paste("cells_fdr0.001", sum(keep))), file.path(out, "summary.tsv"))
